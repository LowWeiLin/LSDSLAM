import java.io.File;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import Utils.Constants;


public class LiveSLAMWrapper {
	
	boolean isInitialized = false;
	int imageSeqNumber = 0;
	LSDSLAM lsdSlam = new LSDSLAM();
	
	final String imagesDirectory = "images";
	
	public LiveSLAMWrapper() {
	}
	
	public void loop() {
		
		// Use images in images directory
		String currentDir = System.getProperty("user.dir");
		final File folder = new File(currentDir + "/" + imagesDirectory);
		
		for (final File fileEntry : folder.listFiles()) {
	        if (fileEntry.isDirectory()) {
	        } else {
	        	System.out.println(imagesDirectory + "/" + fileEntry.getName());
	            Mat frame = Highgui.imread(imagesDirectory + "/" + fileEntry.getName());
	            
	            if (frame.empty()) {
	            	System.out.println("Invalid image frame");
	            	continue;
	            }
	            
				// Process image frame
				newImageCallback(frame);
	        }
	    }
		
	}
	
	public void newImageCallback(Mat image) {
		System.out.println("-------\nFrame " + imageSeqNumber + "\n-------");
		
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
			
			// TODO: remove? call in another thread?
			// Do here, sequentially for now.
			lsdSlam.doMappingIteration();
			
		}
		
	}
	

	public static void main(String[] args) {
		// Load OpenCV native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Set Camera parameters
		Constants.setK(748.000000, 748.000000, 319.000000, 239.000000);
		
		
		LiveSLAMWrapper liveSlamWrapper = new LiveSLAMWrapper();
		liveSlamWrapper.loop();
		
	}
	
}
