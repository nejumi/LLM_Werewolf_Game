#!/bin/bash

while true
do
    # ムゲン出現のための乱数生成 (0-99)
    mugen_random=$((RANDOM % 100))
    # 多言語バージョンのための乱数生成 (0-99)
    multi_lingual_random=$((RANDOM % 100))

    # オプションの初期化
    options=""

    # ムゲン出現 (5%の確率)
    if [ $mugen_random -lt 5 ]; then
        echo "ムゲンが登場します！"
        options+=" --mugen"
    else
        echo "ムゲンは登場しません。"
    fi

    # 多言語バージョン (20%の確率)
    if [ $multi_lingual_random -lt 20 ]; then
        echo "多言語バージョンが発動します！"
        options+=" --multi_lingual"
    else
        echo "多言語バージョンは発動しません。"
    fi

    # プログラムの実行
    python3 llm_werewolf.py$options

    echo "プログラムが終了しました。5秒後に再起動します..."
    sleep 5
done