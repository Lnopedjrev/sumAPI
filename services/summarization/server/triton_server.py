import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor, DynamicBatcher
from pytriton.triton import Triton, TritonConfig
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

triton_config = TritonConfig(
    http_port=os.getenv("HTTP_PORT", 8005)
)


@batch
def infer_fn(**inputs: np.ndarray):
    try:
        (texts, ) = inputs.values()
        flattened_texts = texts.astype("U").flatten()
        inputs = tokenizer(flattened_texts.tolist(), return_tensors="pt").to(model.device)
        outputs = model.generate(input_ids=inputs.input_ids,
                                 attention_mask=inputs.attention_mask,
                                 max_new_tokens=150)
        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_bytes = np.char.encode(generated, 'utf-8')
    except Exception as e:
        print(f"Error during inference: {e}")
        generated_bytes = np.char.encode(["Error during inference", "Error during inference"], "utf-8")
    return [generated_bytes]


def main(config: ModelConfig = None):
    with Triton(config=triton_config) as triton:
        triton.bind(
            model_name="summarization_model",
            infer_func=infer_fn,
            inputs=[Tensor(name="input_text", dtype=bytes, shape=(1,))],
            outputs=[Tensor(name="output_text", dtype=bytes, shape=(1, ))],
            config=config
        )
        print("Starting Triton server...")
        triton.serve()


if __name__ == "__main__":

    model = None
    tokenizer = None
    MODEL_REPO_DIR = os.getenv("MODEL_REPO_DIR", "model_repository/")
    MODEL_NAME = os.getenv("MODEL_NAME", "AntonV/mamba2-1.3b-hf")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                              cache_dir=MODEL_REPO_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 cache_dir=MODEL_REPO_DIR)
    model = model.half().to("cuda")

    model_config = ModelConfig(
        max_batch_size=os.getenv("MAX_BATCH_SIZE", 4),
        batcher=DynamicBatcher(
                    max_queue_delay_microseconds=2000
                )
    )
    main()
