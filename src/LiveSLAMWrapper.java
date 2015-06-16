import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;


public class LiveSLAMWrapper {
	
	boolean isInitialized = false;
	int imageSeqNumber = 0;
	LSDSLAM lsdSlam = new LSDSLAM();
	
	public LiveSLAMWrapper() {
	}
	
	public void loop() {
		while(true) {
			// TODO: Get image here
			
			// Process image frame
			newImageCallback(null);
		}
	}
	
	public void newImageCallback(Mat image) {
		// Increment image sequence number
		imageSeqNumber++;

		// Convert image to grayscale, if necessary
		Mat grayImg;
		if (image.channels() == 1) { 
			grayImg = image;
		} else {
			grayImg = new Mat();
			Imgproc.cvtColor(image, grayImg, Imgproc.COLOR_RGB2GRAY);
		}

		// Assert that we work with 8 bit images
		assert(grayImg.elemSize() == 1);

		// need to initialize
		if(!isInitialized) {
			lsdSlam.randomInit(grayImg, 1);
			isInitialized = true;
		} else if(isInitialized && lsdSlam != null) {
			lsdSlam.trackFrame(grayImg, imageSeqNumber);
		}
	}
	
	
}
