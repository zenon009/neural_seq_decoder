python ./scripts/eval_competition.py \
--modelPath="/vol/fastdata/wrb15144/decoder_data/BrainTranslationCompetition/logs/speechBaseline4/modelWeights" \
--seqLen=150 \
--maxTimeSeriesLen=1200 \
--batchSize=64 \
--datasetPath="/vol/fastdata/wrb15144/decoder_data/BrainTranslationCompetition/ptDecoder_ctc"
