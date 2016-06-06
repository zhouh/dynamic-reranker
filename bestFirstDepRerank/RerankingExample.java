package bestFirstDepRerank;

import edu.stanford.nlp.util.CoreMap;
import nndep.DependencyTree;

/**
 * Created by zhouh on 16-1-11.
 */
public class RerankingExample {
    CoreMap sent;
    DependencyTree goldTree;

    public RerankingExample(DependencyTree goldTree, CoreMap sent) {
        this.goldTree = goldTree;
        this.sent = sent;
    }
}
