package DataStructures;

import g2o.g2o_RobustKernel;
import jeigen.DenseMatrix;
import LieAlgebra.SIM3;

public class KFConstraintStruct
{
	public KFConstraintStruct()
	{
		firstFrame = secondFrame = null;
		information = DenseMatrix.zeros(7, 7);
		//robustKernel = 0;
		//edge = 0;

		usage = meanResidual = meanResidualD = meanResidualP = 0;
		reciprocalConsistency = 0;


		idxInAllEdges = -1;
	}

	public Frame firstFrame;
	public Frame secondFrame;
	public SIM3 secondToFirst;
	public DenseMatrix information;
	public g2o_RobustKernel robustKernel;
	public EdgeSim3 edge;

	public float usage;
	public float meanResidualD;
	public float meanResidualP;
	public float meanResidual;

	public float reciprocalConsistency;

	public int idxInAllEdges;
	
};