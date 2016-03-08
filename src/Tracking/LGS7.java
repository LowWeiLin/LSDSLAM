package Tracking;

import jeigen.DenseMatrix;

public class LGS7 {
	
	// 7x7
	DenseMatrix A;
	
	// 7x1
	DenseMatrix b;

	float error;
	int num_constraints;

	
	void initializeFrom(LGS6 ls6, LGS4 ls4)
	{
	  	A = new DenseMatrix(new double[][]{
	  			{ls6.A.get(0,0), ls6.A.get(0,1), ls6.A.get(0,2), ls6.A.get(0,3), ls6.A.get(0,4), ls6.A.get(0,5), 0},
	  			{ls6.A.get(1,0), ls6.A.get(1,1), ls6.A.get(1,2), ls6.A.get(1,3), ls6.A.get(1,4), ls6.A.get(1,5), 0},
	  			{ls6.A.get(2,0), ls6.A.get(2,1), ls6.A.get(2,2), ls6.A.get(2,3), ls6.A.get(2,4), ls6.A.get(2,5), 0},
	  			{ls6.A.get(3,0), ls6.A.get(3,1), ls6.A.get(3,2), ls6.A.get(3,3), ls6.A.get(3,4), ls6.A.get(3,5), 0},
	  			{ls6.A.get(4,0), ls6.A.get(4,1), ls6.A.get(4,2), ls6.A.get(4,3), ls6.A.get(4,4), ls6.A.get(4,5), 0},
	  			{ls6.A.get(5,0), ls6.A.get(5,1), ls6.A.get(5,2), ls6.A.get(5,3), ls6.A.get(5,4), ls6.A.get(5,5), 0},
	  			{0,0,0,0,0,0,0},
	  	});
	  	
	  	b = new DenseMatrix(new double[][]{
	  			{ls6.b.get(0, 0)},
	  			{ls6.b.get(1, 0)},
	  			{ls6.b.get(2, 0)},
	  			{ls6.b.get(3, 0)},
	  			{ls6.b.get(4, 0)},
	  			{ls6.b.get(5, 0)},
	  			{0} 
	  	});

	  	// add ls4
	  	int[] remap = {2,3,4,6};
	  	for(int i=0;i<4;i++)
	  	{
	  		int idx = remap[i];
	  		b.set(idx, 0, b.get(idx, 0) + ls4.b.get(i, 0));;
	  		for(int j=0;j<4;j++) {
	  			A.set(idx, idx, A.get(idx, idx) + ls4.A.get(i, j));
	  		}
	  	}

	  	num_constraints = ls6.numConstraints + ls4.num_constraints;
	}
	
}
