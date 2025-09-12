call D:\Anaconda\Scripts\activate.bat

%data




%train

set epoch=100
set dataset=BP
set sampleTimes=8

python train.py --Aug empty --dataset %dataset%  --epoch %epoch% --sampleTimes %sampleTimes%
python train.py --Aug uniform --dataset %dataset%  --epoch %epoch% --sampleTimes %sampleTimes%
python train.py --Aug pos --dataset %dataset%  --epoch %epoch% --sampleTimes %sampleTimes%
python train.py --Aug orbit --dataset %dataset%  --epoch %epoch% --sampleTimes %sampleTimes%
python train.py --Aug group --dataset %dataset%  --epoch %epoch% --sampleTimes %sampleTimes%




