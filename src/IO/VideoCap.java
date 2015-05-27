package IO;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.VideoCapture;

public class VideoCap {
    static{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private VideoCapture cap;

    public VideoCap(){
        cap = new VideoCapture();
        
        // Set width/height?
        //cap.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, 640);
        //cap.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, 480);
        
        cap.open(0);
        
        if (!cap.isOpened()) {
        	System.err.println("Failed to open VideoCapture.");
        }
    }
 
    public Mat getMatFrame() {
    	Mat mat = new Mat();
        cap.read(mat);
        return mat;
    }
    
    public void close() {
    	if (cap.isOpened()) {
    		cap.release();
    	}
    }
    
    
}