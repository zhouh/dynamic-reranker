package bestFirstDepRerank;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import nndep.ArcStandard;
import nndep.Config;
import nndep.HierarchicalDepState;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedList;
import java.util.List;
import java.util.Stack;


public class DepRCNNTree{

    /** for leaves this contains reference to a global leaf vector */
    public INDArray vector;
    public INDArray convolutionEmb;
    public INDArray headEmb = null;
    /** gradient is non-null only for leaf nodes. Contains reference. Contains
     * reference to the unique gradient vector corresponding to the leaf */
    private final String label;
    public List<DepRCNNTree> leftChildren;
    public List<DepRCNNTree> rightChildren;
    private int numChild;
    public String head;
    public String leftWord;
    public String rightWord;
    public INDArray leftWordTanhEmb = null;
    public INDArray rightWordTanhEmb = null;
    public INDArray poolingRecordArray = null;
    public INDArray concateEmb = null;
    public INDArray disEmbTanh = null;
    public String disInfo = null;


    public DepRCNNTree(String head, String label, List<DepRCNNTree> leftChildren,
                       List<DepRCNNTree> rightChildren, String leftWord, String rightWord) {
        this.head = head;
        this.label = label;
        this.leftChildren = leftChildren;
        this.rightChildren = rightChildren;
        this.numChild = leftChildren.size() + rightChildren.size();
        this.leftWord = leftWord;
        this.rightWord = rightWord;
    }

    public void setConvolutionEmb(INDArray convolutionEmb) {
        this.convolutionEmb = convolutionEmb;
    }

    public INDArray getConvolutionEmb() {
        return convolutionEmb;
    }

    public String getWord(){
        return head;
    }

    public int numChildren() {
        return leftChildren.size() + rightChildren.size();
    }

    public INDArray getVector() {
        return this.vector;
    }

    public void setVector(INDArray vector) {
        this.vector = vector;
    }

    public String getLabel() {
        return this.label;
    }

    public String getStr(){

        return null;

    }

    public static String getWord(int i, String[] words){

        if(i < 0)
            return "<LEFT>";
        else if (i >= words.length) {
            return  "<RIGHT>";
        }
        else
            return words[i];

    }

    /**
     * return the reranking tree for RCNN
     *
     * @param sent
     * @param state
     * @return
     */
    public static DepRCNNTree HierarchyDepState2RerankingTree(CoreMap sent, HierarchicalDepState state, DepRCNNModel model){

        List<CoreLabel> tokens = sent.get(CoreAnnotations.TokensAnnotation.class);

        // prepare the word and tag sequences
        String[] words = new String[tokens.size()];
        String[] tags = new String[tokens.size()];
        DepRCNNTree[] queue = new DepRCNNTree[tokens.size()];

        for (int i = 0; i < tokens.size(); i++) {
            words[i] = tokens.get(i).word();
            tags[i] = tokens.get(i).tag();
        }

        for (int i = 0; i < tokens.size(); i++) {
            DepRCNNTree node = new DepRCNNTree(words[i], tags[i], new LinkedList<>(),
                    new LinkedList<>(), getWord(i-1, words), getWord(i+1, words));

            node.leftWordTanhEmb = model.getNonlinearEmb(node.leftWord);
            node.rightWordTanhEmb = model.getNonlinearEmb(node.rightWord);
            node.headEmb = model.getNonlinearEmb(node.head);

            queue[i] = node;
        }

        // get the initial state and
        // save all the states in the List
        List<HierarchicalDepState> states = new LinkedList<>();

        while(state != null){
            states.add(0, state);
            state = state.lastState;
        }


        Stack<DepRCNNTree> stack = new Stack<>();
        DepRCNNTree rootTree = new DepRCNNTree(Config.ROOT, Config.ROOT, new LinkedList<>(),
                new LinkedList<>(), Config.ROOTLEFT, Config.ROOTRIGHT);
        rootTree.leftWordTanhEmb = model.getNonlinearEmb(rootTree.leftWord);
        rootTree.rightWordTanhEmb = model.getNonlinearEmb(rootTree.rightWord);
        rootTree.headEmb = model.getNonlinearEmb(rootTree.head);

        stack.push(rootTree);


        for (int i = 1; i < states.size(); i++) {
            HierarchicalDepState currentState = states.get(i);
            HierarchicalDepState lastState = states.get(i - 1);

            if(currentState.actType == ArcStandard.shiftActTypeID){
                // in the buffer, the first word is 0 : root
                int queueTopIndex = lastState.c.getBuffer(0) - 1;
                stack.push(queue[queueTopIndex]);
            }
            else if(currentState.actType == ArcStandard.leftReduceActTypeID){
                DepRCNNTree right = stack.pop();
                DepRCNNTree left = stack.pop();
                right.leftChildren.add(left);

                stack.push(right);

            }
            else if(currentState.actType == ArcStandard.rightReduceActTypeID){
                DepRCNNTree right = stack.pop();
                DepRCNNTree left = stack.pop();
                left.rightChildren.add(right);

                stack.push(left);
            }
        }

        if(stack.size() != 1)
            throw new RuntimeException("Tree Construction Error!");

        return stack.pop();

    }

}