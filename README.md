# A Search-Based Dynamic Reranking Model for Dependency Parsing

------

This is the code for paper "A Search-Based Dynamic Reranking Model for Dependency Parsing".
The parsing system is 

## Reguired Software
 * [Stanford-Corenlp Library](http://stanfordnlp.github.io/CoreNLP/)
 * [nd4j-0.4-rc3.8](https://github.com/deeplearning4j/nd4j/releases/tag/nd4j-0.4-rc3.8)
 * Java8

Note that we use the 0.4-rc3.8 version of ND4J. There are some bugs in that version and we fix them in our codes. We don't whether they fix these bugs in their new release. If true, our fix may be bugs if the new version is used in our code.


## Data Format
We adopt the [CoNLL](http://www.aclweb.org/anthology/D/D07/D07-1096.pdf) data format for dependency parsing. Ten-folds jackknifing is adopted for training the baseline parser, namely ten baseline parsing models should be trained, which will be loaded into memory for training dynamically.

## Training

    datadir=/home/Data
    depDir=$datadir/dep/conll

    java -Xmx15g -cp ./depReranker.jar bestFirstDepRerank.BestFirstDepReranker -trainFile $depDir/train.txt -trainDir $depDir/folds/ -devFile $depDir/devr.txt -embedFile $datadir/embedding.en.25 -batchSize 100 -trainingThreads 7 -model ./model.txt.gz -baseModel $depDir/model.txt.gz -wordCutOff 2 -bUseBeamSample true -nBeam 4 -dMargin 1 -nDisNum 30 -earlyUpdate false -hcScore true -bUseBeamRNNSearch true -nMaxReviseActNum 8

## Parsing

    java -Xmx15g -cp ./depReranker.jar bestFirstDepRerank.BestFirstDepReranker -testFile $depDir/testr.txt -baseModel $depDir/model.txt.gz -model ./model.txt.gz -output ./test.result -bUseBeamSample true -hcScore true -bUseBeamRNNSearch true -nMaxReviseActNum 8 -dMargin 0.999 -nBeam 4 -nOracleDepth 3

Here `-dMargin` is a parameter to control the selections of revising candidates, which will speed up the dynamic reranker significantly without lossing accuracy.

------


[1]: Hao Zhou, Yue Zhang, Shujian Huang, Junsheng Zhou, XIN-YU DAI and Jiajun Chen. A Search-Based Dynamic Reranking Model for Dependency Parsing. In Proceeding of ACL 2016.
