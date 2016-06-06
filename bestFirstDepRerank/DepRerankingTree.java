package bestFirstDepRerank;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import nndep.ArcStandard;
import nndep.Config;
import nndep.HierarchicalDepState;
import org.nd4j.linalg.api.ndarray.INDArray;
import recursivenetwork.util.SumOfGradient;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

import static java.lang.System.exit;

public class DepRerankingTree{

    /** for leaves this contains reference to a global leaf vector */
    private INDArray vector;
    /** INDArray before non-linearity is applied */
    private INDArray preOutput;
    /** gradient is non-null only for leaf nodes. Contains reference. Contains
     * reference to the unique gradient vector corresponding to the leaf */
    private SumOfGradient gradient;
    private final String label;
    private final List<DepRerankingTree> children;
    private final int numChild;
    private String head;
    public boolean bLeft;

    /**
     * @param label
     * @param children
     */
    public DepRerankingTree(String label, List<DepRerankingTree> children, String head, boolean bLeft) {
        this.label = label;
        this.children = children;
        this.numChild = children.size();
        this.head = head;
        this.bLeft = bLeft;
    }

    public String getWord(){
        return head;
    }

    public int numChildren() {
        return this.numChild;
    }

    public DepRerankingTree getChild(int i) {
		/* use iterator for trees with large degree */
        return this.children.get(i);
    }

    public Iterator<DepRerankingTree> getChildren() {
        return this.children.iterator();
    }

    public INDArray getVector() {
        return this.vector;
    }

    public void setVector(INDArray vector) {
        this.vector = vector;
    }

    public SumOfGradient getGradient() {
        return this.gradient;
    }

    public void setGradient(SumOfGradient sumOfGradient) {
        this.gradient = sumOfGradient;
    }

    /** accumulate gradient so that after a set of backprop through several trees
     * you can update all the leaf vectors by the sum of gradients. Don't forget to
     * clear the gradients after updating. */
    public void addGradient(INDArray gradient) {
        this.gradient.addGradient(gradient);
    }

    public String getLabel() {
        return this.label;
    }

    public INDArray getPreOutput() {
        return this.preOutput;
    }

    public void setPreOutput(INDArray preOutput) {
        this.preOutput = preOutput;
    }

    public String getStr(){

        return null;

    }

    public static DepRerankingTree HierarchyDepState2RerankingTree(CoreMap sent, HierarchicalDepState state){

        List<CoreLabel> tokens = sent.get(CoreAnnotations.TokensAnnotation.class);

        // get the initial state and
        // save all the states in the List
        List<HierarchicalDepState> states = new LinkedList<>();

        while(state != null){
            states.add(0, state);
            state = state.lastState;
        }

        Stack<DepRerankingTree> stack = new Stack<>();
        DepRerankingTree rootTree = new DepRerankingTree(Config.ROOT,
                new LinkedList<>(), Config.ROOT, false);
        stack.push(rootTree);


        for (int i = 1; i < states.size(); i++) {
            HierarchicalDepState currentState = states.get(i);
            HierarchicalDepState lastState = states.get(i - 1);

            if(currentState.actType == ArcStandard.shiftActTypeID){
                // in the buffer, the first word is 0 : root
                CoreLabel coreLabel = tokens.get(lastState.c.getBuffer(0) - 1);
                DepRerankingTree tree = new DepRerankingTree(coreLabel.tag(),
                        new LinkedList<>(), coreLabel.word(), false);
                stack.push(tree);
            }
            else if(currentState.actType == ArcStandard.leftReduceActTypeID
                    || currentState.actType == ArcStandard.rightReduceActTypeID){
                DepRerankingTree right = stack.pop();
                DepRerankingTree left = stack.pop();
                List<DepRerankingTree> children = new LinkedList<>();
                children.add(left);
                children.add(right);

                DepRerankingTree tree = currentState.actType == ArcStandard.leftReduceActTypeID ?
                        new DepRerankingTree(
                                right.getLabel(),
                                children,
                                null,
                                true)
                        :
                        new DepRerankingTree(
                                left.getLabel(),
                                children,
                                null,
                                false);
                stack.push(tree);

            }
            else{
                exit(0);
            }


        }

        if(stack.size() != 1)
            throw new RuntimeException("Tree Construction Error!");

        return stack.pop();

    }

}