
public class LGS6 {

	// 6x6 matrix
	jeigen.DenseMatrix A;
	
	// 6x1 vector
	jeigen.DenseMatrix b;
	
	double error;
	int numConstraints;
	
	public LGS6() {
	}
	
	public void initialize() {
		// Initialize to zeros
		A = jeigen.Shortcuts.zeros(6, 6);
		b = jeigen.Shortcuts.zeros(6, 1);
		error = 0;
		numConstraints = 0;
	}
	
	// J is a vec6
	public void update(jeigen.DenseMatrix J, float res, float weight) {	
		A = A.add(J.mmul(J.t()).mul(weight));
		b = b.sub(J.mul(res * weight));
		error += res * res * weight;
		numConstraints += 1;
	}
	
	public void finish() {
	    A = A.div(numConstraints);
	    b = b.div(numConstraints);
	    error /= (float) numConstraints;
	}
	
}
