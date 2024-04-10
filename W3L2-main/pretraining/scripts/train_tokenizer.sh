echo "Download dataset"
python train_tokenizer.py download_wikidata --data_path comet24082002/vie_wiki_dataset --saved_path vi_clean_corpus.txt

echo "Train a new Tokenizer"
python train_tokenizer.py train_tokenizer --sp_model_name vi-tokenizer-10k

echo "Merge new tokenizer to Llama-Tokenizer"
python train_tokenizer.py merge_tokenizer \
    --source_tokenizer_dir meta-llama/Llama-2-7b-hf \
    --new_tokenizer_model vi-tokenizer-10k \
    --new_tokenizer_dir Initial-Vi-Llama

echo "Initialize new model, this process may take some time to complete..."
python train_tokenizer.py reinit_model \
    --model_name meta-llama/Llama-2-7b-hf \
    --new_tokenizer_dir Initial-Vi-Llama

