package bestFirstDepRerank;

import nndep.HierarchicalDepState;

/**
 * Created by zhouh on 16-2-21.
 *
 */
public class BeamItem implements Comparable{

    public HierarchicalDepState state;
    public HierarchicalDepState revisedPoint;
    public double score;

    public BeamItem(HierarchicalDepState state, HierarchicalDepState revisedPoint, double score) {
        this.state = state;
        this.revisedPoint = revisedPoint;
        this.score = score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    @Override
    public int compareTo(Object o) {

        BeamItem s = (BeamItem)o;
        int retval = score > s.score ? -1 : (score == s.score ? 0 : 1);
        return retval;
    }
}
