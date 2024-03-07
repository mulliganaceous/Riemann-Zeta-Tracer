package guiDemo;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Image;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.WindowConstants;

import complex.Complex;

/**Window for the Riemann Zeta simulator
 * @author Mulliganaceous
 */
public class RiemannZetaWindow extends JFrame {
	private static final long serialVersionUID = -1;
	private static final int WINDOW_WIDTH = 1280;
	private static final int WINDOW_HEIGHT = 720;
	private static final int SLEEPDUR = 1000/30;
	private static final int INITIAL = 0;
	private static final int SKIPTO = 0;
	RiemannZetaCriticalModel model = new RiemannZetaCriticalModel();
	RiemannZetaPanel mPanel = new RiemannZetaPanel(model);
	RiemannZetaSelection sPanel = new RiemannZetaSelection(model);
	
	public RiemannZetaWindow() {
		super();
		this.setLayout(new BorderLayout());
		this.getContentPane().add(mPanel);
		// this.getContentPane().add(sPanel);
		
		// Set initial value
		this.model.setInterval(1);
		this.model.setS(new Complex(0.5, INITIAL));
		
		// Set visible
		this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		this.pack();
		this.setSize(new Dimension(WINDOW_WIDTH + this.getInsets().left + this.getInsets().right,WINDOW_HEIGHT + this.getInsets().top + this.getInsets().bottom));
		Image icon;
		try {
			icon = ImageIO.read(new File("images/Riemann65536.gif"));
			this.setIconImage(icon);
		} catch (IOException e) {
			e.printStackTrace();
		};
		this.setTitle("Million Dollar Limaçon");
		this.setResizable(true);
		this.setVisible(true);
	}
	
	private void printrecord(boolean printzeros, double[] xheight, int xheightprev, double[] drecord, int drecordprev) {
		int xind = (int) xheight[4];
		int dind = (int) drecord[6];
		
		// Last zero
		String heightstr = "";
		
		if (printzeros && model.localminima() && model.getZetaS().abs() < 1/RiemannZetaCriticalModel.THRESHOLD) {
			double height = model.getHeight();
			if (!Double.isNaN(height)) {
				heightstr = String.format("last height: ~%3.3f\n", height);
				System.out.printf("| ζ(s)|= %-31.3f" + heightstr, model.getZetaS().abs());
			}
			
			heightstr = "";
			double zeropathlength = model.getZeropath();
			heightstr = String.format("arc length: %3.3f\n", zeropathlength);
			System.out.printf("|ζ′(s)|= %-31.3f" + heightstr, model.getSpeed());
			
			heightstr = "";
			double dheight = model.getDHeight();
			double darc = model.getDArc();
			if (!Double.isNaN(dheight)) {
				heightstr = String.format("distance: ~%3.3f (arc: %3.3f)\n", dheight, darc);
				System.out.printf(" zeros \u2248 %-31d" + heightstr, model.getZeroes());
			}
		}

		// Records
		if ((xind & 1) == 1) {
			heightstr = String.format("Closest : Δ \u2248 %3.3E @ h1\u2248 %3.3f\n", xheight[0], xheight[2]);
			System.out.printf(heightstr, model.getZeroes());
		}
		
		heightstr = "";
		if ((xind & 2) == 0 && ((xheightprev & 2) != 0)) {
			heightstr = String.format("Farthest:|ζ|\u2248 %3.3f @ h \u2248 %3.3f\n", xheight[1], xheight[3]);
			System.out.printf(heightstr, model.getZeroes());
		}
		
		heightstr = "";
		if ((dind & 2) != 0) {
			heightstr = String.format("Shortest: L \u2248 %3.3E @ h \u2248 %3.3f\n", drecord[2], drecord[3]);
			System.out.printf(heightstr);
		}
		
		heightstr = "";
		if ((dind & 4) != 0) {
			heightstr = String.format("Longest : L \u2248 %3.3f @ h \u2248 %3.3f\n", drecord[4], drecord[5]);
			System.out.printf(heightstr);
		}
		
		heightstr = "";
		if ((dind & 1) == 0 && (drecordprev & 1) != 0) {
			heightstr = String.format("Fastest : L \u2248 %3.3f @ h \u2248 %3.3f\n", drecord[0], drecord[1]);
			System.out.printf(heightstr);
		}
		
		xheightprev = (int)xheight[4];
		drecordprev = (int)drecord[6];
	}
	
	void skip(double height) {
		double curheight = this.model.getS().im();
		int xheightprev = 0;
		int drecordprev = 0;
		while (curheight < height - RiemannZetaCriticalModel.INCREMENT) {
			if (curheight % 1 == 0)
				System.out.printf("h = %.6f\n", curheight);
			
			double[] xheight = model.getXHeight();
			double[] drecord = model.getDrecord();
			printrecord(true, xheight, xheightprev, drecord, drecordprev);
			xheightprev = (int)xheight[4];
			drecordprev = (int)drecord[6];
			
			model.increment();
			curheight = this.model.getS().im();
		}
		Complex s = model.getS();
		Complex z = model.getZetaS();
		System.out.printf("ζ(%.2f + %.6f\u2148) =\t %.6f %c %.6f\u2148\t(|ζ| = %.6f)\n", s.re(), s.im(), z.re(), z.im() >= 0 ? '+' : '-', Math.abs(z.im()), z.abs());
		model.increment();
	}
	
	void loop() {
		int xheightprev = 0;
		int drecordprev = 0;
		while(true) {
			try {
				Complex s = model.getS();
				Complex z = model.getZetaS();
				if (model.getOffset() % RiemannZetaCriticalModel.INCREMENT_LEVEL == 0)
					System.out.printf("ζ(%.2f + %.6f\u2148) =\t %.6f %c %.6f\u2148\t(|ζ| = %.6f)\n", s.re(), s.im(), z.re(), z.im() >= 0 ? '+' : '-', Math.abs(z.im()), z.abs());
				
				double[] xheight = model.getXHeight();
				double[] drecord = model.getDrecord();
				printrecord(false, xheight, xheightprev, drecord, drecordprev);
				xheightprev = (int)xheight[4];
				drecordprev = (int)drecord[6];
				
				// Increment
				model.increment();
			} catch (Exception exception) {
				exception.printStackTrace();
			}
		}
	}
	
	public static void main(String[] args) {
		RiemannZetaWindow window = new RiemannZetaWindow();
		window.skip(INITIAL + SKIPTO);
		DrawPanel th1 = new DrawPanel(window);
		try {
			for (int k = 30; k > 0; k--) {
				System.out.printf("\rCountdown: %2d", k);
				Thread.sleep(1000);
			}
			System.out.println();
			window.mPanel.saveImage(INITIAL + SKIPTO);
			window.mPanel.repaint();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		th1.start();
		window.loop();
	}
	
	static class DrawPanel implements Runnable {
		Thread t;
		RiemannZetaWindow window;
		public DrawPanel(RiemannZetaWindow window) {
			this.window = window;
		}
		@Override
		public void run() {
			while(true) {
				try {
					window.mPanel.repaint();
					Thread.sleep(SLEEPDUR);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
		
		public void start() {
			System.out.println("Starting animator\n*****************");
			if (t == null) {
				t = new Thread(this);
				t.start();
			}
		}
	}
}