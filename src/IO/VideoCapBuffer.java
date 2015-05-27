package IO;
import java.util.LinkedList;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;


/**
 * Reads video frames and holds in a buffer.
 */
public class VideoCapBuffer {

	private VideoCap videoCap;
	private CaptureThread capThread;
	private boolean captureFlag = false;
	
	public LinkedList<Mat> frameRGBBuffer;
	public LinkedList<Mat> frameGrayBuffer;
	
	public VideoCapBuffer() {
		frameRGBBuffer = new LinkedList<Mat>();
		frameGrayBuffer = new LinkedList<Mat>();
		startCapture();
	}
	
	public void startCapture() {
		captureFlag = true;
		videoCap = new VideoCap();
		
		capThread = new CaptureThread();
		capThread.setDaemon(true);
		capThread.start();
	}
	
	public void stopCapture() {
		captureFlag = false;
		videoCap.close();
	}
	
	class CaptureThread extends Thread {
		@Override
		public void run() {
			while (captureFlag) {
				// Read RGB frame
				Mat frame = videoCap.getMatFrame();
				
				// Convert to gray scale
				Mat grayFrame = new Mat();
				Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_RGB2GRAY);

				// Add RGB frame to buffer
				frameRGBBuffer.add(frame);
				
				// Add gray frame to buffer
				// TODO: remove limit
				if (frameGrayBuffer.size() < 10) {
					frameGrayBuffer.add(grayFrame);
				}
				
			}
		}
	}
}
