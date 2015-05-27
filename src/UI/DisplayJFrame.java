package UI;
import java.awt.Graphics;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.util.LinkedList;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

import org.opencv.core.Mat;

import IO.Mat2Image;

@SuppressWarnings("serial")
public class DisplayJFrame extends JFrame implements KeyListener, WindowListener{
	private JPanel contentPanel;
	private Mat2Image mat2Image = new Mat2Image();
	
	public LinkedList<Mat> frameBuffer;
	
	private RepaintThread repaintThread;
	
	/**
	 * Create the frame.
	 */
	public DisplayJFrame() {
		
		setBounds(100, 100, 650, 490);
		contentPanel = new JPanel();
		contentPanel.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPanel);
		contentPanel.setLayout(null);
		
		addKeyListener(this);
		addWindowListener(this);

		repaintThread = new RepaintThread();
		repaintThread.setDaemon(true);
		repaintThread.start();
	}


	public void paint(Graphics g) {
		g = contentPanel.getGraphics();
		
		System.out.println(frameBuffer.size());
		
		// Get buffered frame
		Mat frame = frameBuffer.poll();
		if (frame == null) {
			return;
		}
		
		// Convert to BufferedImage
		BufferedImage image = mat2Image.getImage(frame);
		
		
		// Draw BufferedImage
		g.drawImage(image, 0, 0, this);
		
	}

	class RepaintThread extends Thread {
		@Override
		public void run() {
			while (true) {
				repaint();
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
				}
			}
		}
	}

	
	// =============
	// Key Listeners
	// =============
	
	@Override
	public void keyPressed(KeyEvent e) {
	}

	@Override
	public void keyReleased(KeyEvent e) {
		if(e.getKeyCode() == KeyEvent.VK_ESCAPE) {
            this.dispatchEvent(new WindowEvent(this, WindowEvent.WINDOW_CLOSING));
		}
	}

	@Override
	public void keyTyped(KeyEvent e) {
	}
	
	// ================
	// Window Listeners
	// ================

	@Override
	public void windowActivated(WindowEvent e) {
	}

	@Override
	public void windowClosed(WindowEvent e) {
	}

	@Override
	public void windowClosing(WindowEvent e) {
		System.out.println("exit");
		System.exit(0);
	}

	@Override
	public void windowDeactivated(WindowEvent e) {
	}

	@Override
	public void windowDeiconified(WindowEvent e) {
	}

	@Override
	public void windowIconified(WindowEvent e) {
	}

	@Override
	public void windowOpened(WindowEvent e) {
		
	}
}