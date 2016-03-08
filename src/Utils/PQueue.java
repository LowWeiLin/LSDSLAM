package Utils;

import java.util.PriorityQueue;

public class PQueue<T> {

    private PriorityQueue<IntPriorityComparableWrapper<T>> queue;

    public PQueue() {
        queue = new java.util.PriorityQueue<IntPriorityComparableWrapper<T>>();
    }

    public void add( int priority, T object ) {
        queue.add( new IntPriorityComparableWrapper<T>(object, priority) );
    }

    public T get() {
        return (null != queue.peek())? queue.poll().getObject() : null;
    }
    
    public int peekPriority() {
		return queue.peek().priority;
    }
    
    public boolean isEmpty() {
    	return queue.isEmpty();
    }


    /**
     * A "wrapper" to impose comparable properties on any object placed in the
     * queue.
     */
    private static class IntPriorityComparableWrapper<T>
    implements Comparable<IntPriorityComparableWrapper<T>> {

        private T object;
        private int priority;

        public IntPriorityComparableWrapper( T object, int priority ) {
            this.object = object;
            this.priority = priority;
        }

        public int compareTo( IntPriorityComparableWrapper<T> anotherObject ) {
            return this.priority - anotherObject.priority;
        }

        public int getPriority() {
            return priority;
        }

        public T getObject() {
            return object;
        }
    }

}