
net=vit_base_patch16_224.orig_in21k_ft_in1k


# #### Settings for IBA
# method=iba
# st=4
# et=5
# op=blocks
# beta=10
# save=${net}_${method}_st${st}_et${et}_op${op}_beta${beta}


#### Settings for COIBA
method=coiba
st=4
et=12
op=norm1
beta=1
save=${net}_${method}_st${st}_et${et}_op${op}_beta${beta}


python generate.py \
  $method \
  --data-path data/imagenet \
  --start-target $st --end-target $et \
  --op-name $op \
  --network $net \
  --beta $beta \
  --save-path $save \
  --load_estim \