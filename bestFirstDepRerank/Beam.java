package bestFirstDepRerank;

import java.util.Comparator;
import java.util.concurrent.PriorityBlockingQueue;

/**
 * Created by zhouh on 16-2-21.
 *
 */
public class Beam extends PriorityBlockingQueue<BeamItem> {

    private static final long serialVersionUID = 1L;

    public int beam;

    /**
     * the constructor of the new priority queue
     */
    public Beam(int beam){

        //the new comparator function
        super(1,new Comparator<BeamItem>(){
            //			@Override
            public int compare(BeamItem o1, BeamItem o2) {
                if( o1.score < o2.score)
                    return -1 ;
                else if(o1.score == o2.score){
                    return 0;
                }else{
                    return 1;
                }
            }
        });

        this.beam=beam;

    }

    /**
     * insert a new item into the chart
     * the chart always keep the best beam item in the chart
     * @param item
     */
    public void insert(BeamItem item){

        if(this.size()<beam) {
            offer(item);
        }
        else if(item.score<=peek().score) return;
        else {
            poll();
            offer(item);
        }
    }

    public void clearAll() {
        this.clear();
    }

}
