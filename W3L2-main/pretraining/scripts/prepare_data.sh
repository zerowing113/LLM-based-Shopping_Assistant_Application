echo "Preparing streaming dataset..."
python prepare_streaming_dataset.py \
    --path comet24082002/vie_wiki_dataset \
    --out_root vi-wiki \
    --tokenizer Initial-Vi-Llama