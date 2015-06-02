package DataStructures;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import Utils.Constants;


public class Frame {

	public Mat[] imageLvl;
	public byte[][] imageArrayLvl; // Array of image data for fast reading
	
	public Frame(Mat image) {
		
		this.imageLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageArrayLvl = new byte[Constants.PYRAMID_LEVELS][];
		
		this.imageLvl[0] = image;
		this.imageArrayLvl[0] = new byte[(int) imageLvl[0].total()];
		this.imageLvl[0].get(0, 0, imageArrayLvl[0]);
		
		// Generate lower levels
		for (int i=1 ; i<Constants.PYRAMID_LEVELS ; i++) {
			this.imageLvl[i] = new Mat();
			Imgproc.pyrDown(this.imageLvl[i-1], this.imageLvl[i]);
			this.imageArrayLvl[i] = new byte[(int) imageLvl[i].total()];
			this.imageLvl[i].get(0, 0, imageArrayLvl[i]);
		}
	}
	
	public void getGradient() {
		
	}
	
//	public int width() {
//		return imageLvl[0].width();
//	}
//	
//	public int height() {
//		return imageLvl[0].height();
//	}
	
	public int width(int level) {
		return imageLvl[level].width();
	}
	
	public int height(int level) {
		return imageLvl[level].height();
	}
}
