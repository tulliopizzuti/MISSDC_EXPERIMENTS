#!/bin/bash

#dset_name=$1

# XE Loss

for dset_name in "adultsample10" "australian" "contraceptive" "credit" "flare" "fodorszagats" "imdb" "mammogram" "tax5000trimmed" "thoracic" "tictactoe"
#for dset_name in  'imdb'
do
clean_dataset=data/clean/$dset_name.csv
for p in "05" "20" "50"
do
  f=data/dirty/$dset_name'_allcolumns_'$p'.csv'
  for emb in   "ft" "embdi_f4" # "--text_emb data/pretrained-emb/""$dset_name""_""$p""_embdi_f4.emb"
#  for emb in  "--text_emb data/pretrained-emb/""$dset_name""_""$p""_ft.emb" "--text_emb data/pretrained-emb/""$dset_name""_""$p""_embdi_f4.emb"
  do
    for ncol in  {1..2}
    do
    echo "###########################" $f
    python main_multilabel.py --ground_truth $clean_dataset \
    --dirty_dataset $f \
    --architecture multitask --graph_layers 2 --gnn_feats 64 --h_feats 128 --predictor_layers 2 --head_layers 2 \
    --training_subset target --loss xe  --epochs 300 --grace 150 --dropout_clf 0.2 \
    --text_embs "data/pretrained-emb/""$dset_name""_""$p"_"$emb".emb --max_components 64 --head_model linear --shared_model linear --save_imputed_df --imputed_df_tag $emb
   done
  done
  for ncol in  {1..2}
   do
    echo "###########################" $f
    python main_multilabel.py --ground_truth $clean_dataset \
    --dirty_dataset $f \
    --architecture multitask --graph_layers 2 --gnn_feats 64 --h_feats 128 --predictor_layers 2 --head_layers 2 \
    --training_subset target --loss xe  --epochs 300 --grace 150 --dropout_clf 0.2 \
    --text_embs data/pretrained-emb/"$dset_name"_"$p"_ft.emb data/pretrained-emb/"$dset_name"_"$p"_embdi_f4.emb \
    --max_components 64 --head_model linear --shared_model linear --save_imputed_df --imputed_df_tag "fd_embdi"

  done
done
done
