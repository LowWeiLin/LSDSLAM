package Tracking;

import jeigen.DenseMatrix;

public class LGS4
{

	DenseMatrix A;	// 4x4
	DenseMatrix b;		// 4x1

	float error;
	int num_constraints;

	public LGS4() {
		initialize();
	}
	
	void initialize()
	{
		A = jeigen.Shortcuts.zeros(4, 4);
		b = jeigen.Shortcuts.zeros(4, 1);
		//memset(SSEData,0, sizeof(float)*4*15);
		error = 0;
		this.num_constraints = 0;
	}

	void finishNoDivide()
	{
	}

	// J - 4x1
	void update(DenseMatrix J, float res, float weight)
	{
		A = A.add(J.mmul(J.t()).mul(weight));
		b = b.sub(J.mul(res * weight));
		error += res * res * weight;
		num_constraints += 1;
	}

	//float[] SSEData = new float[4*15];
}