#!/bin/bash

DIR=.

for pathfile in $DIR/exp_sotsuron*.sh; do
    pjsub $pathfile
    break  # 実際の実行時はコメントアウトする
done
