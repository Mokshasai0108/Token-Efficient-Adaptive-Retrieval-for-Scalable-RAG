"""
TEAR — Module 9: LLM Generation
Uses Llama-3-70B-Instruct with 4-bit quantization via bitsandbytes.
Accepts optimized context from the TEAR pipeline and generates answers.
"""

import time
import torch
from typing import Optional
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from loguru import logger


@dataclass
class GenerationResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    model_id: str


class LLMGenerator:
    """
    Llama-3-70B-Instruct generation module.

    Features:
    - 4-bit NF4 quantization (fits ~40GB VRAM or CPU offload)
    - Chat template formatting (Llama-3 instruct format)
    - Token usage tracking
    - Streaming support
    """

    SYSTEM_PROMPT = """You are a precise, knowledgeable assistant.
Answer the question using ONLY the provided passages.
Be concise and factual. If the answer is not in the passages, say "I cannot find this in the provided context."
Do not hallucinate or add information beyond what is given."""

    def __init__(self, config):
        self.config = config
        self.model_id = config.llm_model_id
        self._load_model()

    def _load_model(self):
        """Load Llama-3-70B with 4-bit quantization."""
        logger.info(f"Loading LLM: {self.model_id}")
        logger.info("Using 4-bit NF4 quantization (bitsandbytes)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.llm_load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,         # nested quantization
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.config.hf_token,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.config.llm_device_map,
            token=self.config.hf_token,
            trust_remote_code=True,
        )

        self.model.eval()
        logger.info("LLM loaded successfully")

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate an answer given a query and compressed context.

        Args:
            query:          the user's question
            context:        assembled context from TEAR pipeline
            max_new_tokens: override config setting

        Returns:
            GenerationResult with answer and token stats
        """
        max_tokens = max_new_tokens or self.config.llm_max_new_tokens

        # Build Llama-3 chat-formatted prompt
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    f"Answer:"
                )
            }
        ]

        # Apply Llama-3 chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        prompt_token_count = inputs["input_ids"].shape[1]

        # Generate
        start = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.config.llm_temperature,
                do_sample=self.config.llm_temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        latency = time.time() - start

        # Decode only new tokens (strip prompt)
        new_token_ids = output_ids[0][prompt_token_count:]
        answer = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        completion_tokens = len(new_token_ids)

        return GenerationResult(
            answer=answer,
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_tokens,
            total_tokens=prompt_token_count + completion_tokens,
            latency_seconds=round(latency, 3),
            model_id=self.model_id,
        )

    def generate_stream(self, query: str, context: str):
        """
        Streaming generator — yields tokens one-by-one.
        Use this for the FastAPI streaming endpoint.
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.config.llm_max_new_tokens,
            temperature=self.config.llm_temperature,
            do_sample=self.config.llm_temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        import threading
        thread = threading.Thread(
            target=self.model.generate, kwargs=generation_kwargs
        )
        thread.start()

        for token in streamer:
            yield token

        thread.join()
