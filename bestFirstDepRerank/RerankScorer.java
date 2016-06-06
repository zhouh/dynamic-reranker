package bestFirstDepRerank;

/**
 * Created by zhouh on 16-1-10.
 *
 * This is the reranking scorer for dependency best-first reranker except the base
 * parser model score
 *
 */
public class RerankScorer {

    public DepRCNN hRNN;
    public DepRCNN cRNN;

    public RerankScorer(DepRCNN hScore, DepRCNN cScore) {
        this.hRNN = hScore;
        this.cRNN = cScore;
    }

    public double getHScore(DepRCNNTree tree){

        return hRNN.getScore(tree);

    }

    public double getCScore(DepRCNNTree tree){

        return cRNN.getScore(tree);

    }
}
