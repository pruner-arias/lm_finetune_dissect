
import hydra
import torch
import inspect
import transformers
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from accelerate import Accelerator
from datasets import load_dataset


@hydra.main(version_base=None, config_path="config")
def main(config: DictConfig) -> None:
    
    # Prepare model
    model = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=torch.bfloat16)

    if 'lora' in config:
        if isinstance(config.lora, str):
            model = PeftModel.from_pretrained(model, config.lora)
        else:
            model = get_peft_model(
                model, 
                LoraConfig(**OmegaConf.to_object(config.lora))
            )
            model.enable_input_require_grads()
    
    model.gradient_checkpointing_enable()

    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer if 'tokenizer' in config else config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id            # patch llama tokenizer

    def tokenize_func(eg):
        result = tokenizer(
            eg["text"] if "text" in eg else eg["MedlineCitation"]["Article"]["Abstract"]["AbstractText"],       # extract abstract text from pubmed corpus
            truncation=True,
            max_length=config.max_token_length,
            padding=False,
            return_tensors=None,
        )
        return result
                                              
    with Accelerator().main_process_first():
        dataset = load_dataset(config.dataset)[config.split]
        if config.subset_examples is not None:
            dataset = dataset.shuffle(seed=config.seed).select(range(config.subset_examples))
        dataset = dataset.map(
            tokenize_func, 
            remove_columns=list(set(dataset.column_names) - set(inspect.signature(model.forward).parameters.keys()))
        )

    # Prepare trainer
    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.accumulate_grad_batches,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.max_epochs,
            gradient_checkpointing=True,
            optim="adamw_torch",
            learning_rate=config.lr,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.scheduler,
            max_grad_norm=config.gradient_clip_val,
            bf16=True,                
            logging_steps=10,
            evaluation_strategy="no",
            eval_accumulation_steps=1,
            save_strategy="no",
            output_dir='tmp',
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            report_to="none",
            seed=config.seed,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8,
        ),
    )

    # Perform training or evaluation
    if config.mode == 'train':
        trainer.train_dataset = dataset
        trainer.train()
        trainer.save_model(config.save_dir)

    elif config.mode == 'eval':
        loss = trainer.evaluate(dataset)['eval_loss']
        print(f'Loss on {config.split}: {loss:.4f}')

    else:
        raise ValueError("Invalid mode: %s" % config.mode)

if __name__ == "__main__":
    main()
