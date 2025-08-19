label="gpu-comparison-remote"
device="gpu"

./resources/automatic-run.sh 5 sql rq3-dec llama-3.1-8B-inst-hf import lama3.1-8 "1 2 3 4 5 6 7 8 9" $label $device
./resources/automatic-run.sh 5 sql rq3-dec llama-3.2-1B-inst-hf import lama3.2-1 "1 2 3 4 5 6 7 8 9" $label $device
./resources/automatic-run.sh 5 sql rq3-dec llama-3.2-3B-inst-hf import lama3.2-3 "1 2 3 4 5 6 7 8 9" $label $device
./resources/automatic-run.sh 5 sql rq3-dec falcon-3-10B-inst-hf import falcon3-10 "1 2 3 4 5 6 7 8 9" $label $device
./resources/automatic-run.sh 5 sql rq3-dec mistral-7B-inst-v0.3-hf import mistral-7 "1 2 3 4 5 6 7 8 9" $label $device
#./resources/automatic-run.sh 5 sql rq3-dec gpt api gpt "1 2 3 4 5 6 7 8 9" $label

./resources/automatic-run.sh 5 demand rq5 llama-3.1-8B-inst-hf import lama3.1-8 "1 2 3 4 5 6 7 8 9" $label $device
./resources/automatic-run.sh 5 demand rq5 llama-3.2-1B-inst-hf import lama3.2-1 "1 2 3 4 5 6 7 8 9" $label $device
./resources/automatic-run.sh 5 demand rq5 llama-3.2-3B-inst-hf import lama3.2-3 "1 2 3 4 5 6 7 8 9" $label $device
./resources/automatic-run.sh 5 demand rq5 falcon-3-10B-inst-hf import falcon3-10 "1 2 3 4 5 6 7 8 9" $label $device
./resources/automatic-run.sh 5 demand rq5 mistral-7B-inst-v0.3-hf import mistral-7 "1 2 3 4 5 6 7 8 9" $label $device
#./resources/automatic-run.sh 5 demand rq5 gpt api gpt "1 2 3 4 5 6 7 8 9" $label
