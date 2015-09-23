package DepthEstimation;

public class DepthMapPixelHypothesis {


	/** Flag telling if there is a valid estimate at this point.
	 * All other values are only valid if this is set to true. */
	public boolean isValid;

	/** Flag that blacklists a point to never be used - set if stereo fails repeatedly on this pixel. */
	public int blacklisted;

	/** How many frames to skip ahead in the tracked-frames-queue. */
	public float nextStereoFrameMinID;

	/** Counter for validity, basically how many successful observations are incorporated. */
	public int validity_counter;

	/** Actual Gaussian Distribution.*/
	public float idepth;
	public float idepth_var;

	/** Smoothed Gaussian Distribution.*/
	public float idepth_smoothed;
	public float idepth_var_smoothed;
	
	
	
	public DepthMapPixelHypothesis() {
		isValid = false;
		blacklisted = 0;
	}
	
	// Copy constructor
	public DepthMapPixelHypothesis(DepthMapPixelHypothesis p) {
		this.isValid = p.isValid;
		this.blacklisted = p.blacklisted;
		this.nextStereoFrameMinID = p.nextStereoFrameMinID;
		this.validity_counter = p.validity_counter;
		this.idepth = p.idepth;
		this.idepth_var = p.idepth_var;
		this.idepth_smoothed = p.idepth_smoothed;
		this.idepth_var_smoothed = p.idepth_var_smoothed;
	}
	
	public DepthMapPixelHypothesis(float my_idepth,
								   float my_idepth_smoothed,
								   float my_idepth_var,
								   float my_idepth_var_smoothed,
								   int my_validity_counter) {
		isValid = true;
		blacklisted = 0;
		nextStereoFrameMinID = 0;
		validity_counter = my_validity_counter;
		idepth = my_idepth;
		idepth_var = my_idepth_var;
		idepth_smoothed = my_idepth_smoothed;
		idepth_var_smoothed = my_idepth_var_smoothed;
	}
	
	public DepthMapPixelHypothesis(float my_idepth,
								   float my_idepth_var,
								   int my_validity_counter) {
		isValid = true;
		blacklisted = 0;
		nextStereoFrameMinID = 0;
		validity_counter = my_validity_counter;
		idepth = my_idepth;
		idepth_var = my_idepth_var;
		idepth_smoothed = -1;
		idepth_var_smoothed = -1;
	}
	
	// byte[3]
	byte[] getVisualizationColor(int lastFrameID)
	{
		
		float id;
		id= idepth_smoothed;

		if(id < 0)
			return new byte[]{(byte) 255,(byte) 255,(byte) 255};

		// rainbow between 0 and 4
		float r = (float) ((0f-id) * 255f / 1.0f); if(r < 0) r = -r;
		float g = (float) ((1f-id) * 255f / 1.0f); if(g < 0) g = -g;
		float b = (float) ((2f-id) * 255f / 1.0f); if(b < 0) b = -b;

		int rc = (int) (r < 0 ? 0 : (r > 255 ? 255 : r));
		int gc = (int) (g < 0 ? 0 : (g > 255 ? 255 : g));
		int bc = (int) (b < 0 ? 0 : (b > 255 ? 255 : b));

		return new byte[]{(byte) (255-rc),(byte) (255-gc),(byte) (255-bc)};
		/*
		if(debugDisplay == 0 || debugDisplay == 1)
		{
			float id;
			if(debugDisplay == 0)
				id= idepth_smoothed;
			else // if(debugDisplay == 1)
				id= idepth;
	
			if(id < 0)
				return new byte[]{(byte) 255,(byte) 255,(byte) 255};
	
			// rainbow between 0 and 4
			float r = (0-id) * 255 / 1.0; if(r < 0) r = -r;
			float g = (1-id) * 255 / 1.0; if(g < 0) g = -g;
			float b = (2-id) * 255 / 1.0; if(b < 0) b = -b;
	
			uchar rc = r < 0 ? 0 : (r > 255 ? 255 : r);
			uchar gc = g < 0 ? 0 : (g > 255 ? 255 : g);
			uchar bc = b < 0 ? 0 : (b > 255 ? 255 : b);
	
			return new byte[]{255-rc,255-gc,255-bc};
		}
	
		// plot validity counter
		if(debugDisplay == 2)
		{
			float f = validity_counter * (255.0 / (VALIDITY_COUNTER_MAX_VARIABLE+VALIDITY_COUNTER_MAX));
			uchar v = f < 0 ? 0 : (f > 255 ? 255 : f);
			return new byte[]{0,v,v};
		}
	
		// plot var
		if(debugDisplay == 3 || debugDisplay == 4)
		{
			float idv;
			if(debugDisplay == 3)
				idv= idepth_var_smoothed;
			else
				idv= idepth_var;
	
			float var = - 0.5 * log10(idv);
	
			var = var*255*0.333;
			if(var > 255) var = 255;
			if(var < 0)
				return new byte[]{0,0, 255};
	
			return new byte[]{255-var,var, 0};// bw
		}
	
		// plot skip
		if(debugDisplay == 5)
		{
			float f = (nextStereoFrameMinID - lastFrameID) * (255.0 / 100);
			uchar v = f < 0 ? 0 : (f > 255 ? 255 : f);
			return new byte[]{v,0,v};
		}
		*/
		//return new byte[]{(byte) 255,(byte) 255,(byte) 255};
	}
	
}
