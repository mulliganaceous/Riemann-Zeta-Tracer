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
	public static final int WINDOW_WIDTH = 1280;
	public static final int WINDOW_HEIGHT = 720;
	private static final int SLEEPDUR = 1000/144;
	private static final int SLEEPFRAME = 1000/144;
	private static final double INITIAL = 0;
	private static int SKIPTO = 0;
	private static int CHECKPOINT = 256;
	private static int COUNTDOWN = 4;
	private RiemannZetaCriticalModel model = new RiemannZetaCriticalModel();
	private RiemannZetaPanel mPanel = new RiemannZetaPanel(model);
	// private RiemannZetaSelection sPanel = new RiemannZetaSelection(model);
	
	public RiemannZetaWindow() {
		super();
		this.setLayout(new BorderLayout());
		this.getContentPane().add(mPanel);
		// this.getContentPane().add(sPanel);
		
		// Set initial value
		this.model.setInterval(SKIPTO != 0 ? SKIPTO : 1);
		this.model.setS(new Complex(0.5, INITIAL));
		
		// Set visible
		this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		this.pack();
		this.setSize(new Dimension(
				WINDOW_WIDTH /*+ this.getInsets().left + this.getInsets().right*/, 
				WINDOW_HEIGHT /*+ this.getInsets().top + this.getInsets().bottom*/));
		Image icon;
		try {
			icon = ImageIO.read(new File("images/RiemannLogo.gif"));
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
		
		if (printzeros && (model.getForm()[1] & 2) != 0) {
			double height = model.getHeight();
			if (!Double.isNaN(height)) {
				heightstr = String.format("last height: ~%3.3f\n", height);
				System.out.printf("\t| ζ(s)|= %-31.3E" + heightstr, model.getZetaS().abs());
			}
			
			heightstr = "";
			double zeropathlength = model.getZeropath();
			heightstr = String.format("arc length: %3.3f\n", zeropathlength);
			System.out.printf("\t|ζ′(s)|= %-31.3f" + heightstr, model.getSpeed());
			
			heightstr = "";
			double dheight = model.getDHeight();
			double darc = model.getDArc();
			if (!Double.isNaN(dheight)) {
				heightstr = String.format("distance: ~%3.3f (arc: %3.3f)\n", dheight, darc);
				System.out.printf("\t zeros \u2248 %-31d" + heightstr, model.getZeroes());
			}
		}

		// Records
		if ((xind & 1) == 1) {
			heightstr = String.format("\tClosest : Δ \u2248 %3.3E @ h1\u2248 %3.3f\n", xheight[0], xheight[2]);
			System.out.printf(heightstr, model.getZeroes());
		}
		
		heightstr = "";
		if ((xind & 2) == 0 && ((xheightprev & 2) != 0)) {
			heightstr = String.format("\tFarthest:|ζ|\u2248 %3.3f @ h \u2248 %3.3f\n", xheight[1], xheight[3]);
			System.out.printf(heightstr, model.getZeroes());
		}
		
		heightstr = "";
		if ((dind & 2) != 0) {
			heightstr = String.format("\tShortest: L \u2248 %3.3E @ h \u2248 %3.3f\n", drecord[2], drecord[3]);
			System.out.printf(heightstr);
		}
		
		heightstr = "";
		if ((dind & 4) != 0) {
			heightstr = String.format("\tLongest : L \u2248 %3.3f @ h \u2248 %3.3f\n", drecord[4], drecord[5]);
			System.out.printf(heightstr);
		}
		
		heightstr = "";
		if ((dind & 1) == 0 && (drecordprev & 1) != 0) {
			heightstr = String.format("\tFastest : L \u2248 %3.3f @ h \u2248 %3.3f\n", drecord[0], drecord[1]);
			System.out.printf(heightstr);
		}
		
		xheightprev = (int)xheight[4];
		drecordprev = (int)drecord[6];
	}
	
	private void skip(double height, double checkpoint) {
		double curheight = this.model.getS().im();
		int xheightprev = 0;
		int drecordprev = 0;
		final double distance = height - RiemannZetaCriticalModel.INCREMENT;
		while (curheight <= distance) {
			if (curheight % 1 == 0)
				System.out.printf("h = %.6f\n", curheight);
			if (curheight % checkpoint == 0) {
				this.mPanel.saveImage((int)(curheight));
				this.mPanel.repaint();
			}
			
			double[] xheight = model.getXHeight();
			double[] drecord = model.getDrecord();
			printrecord(true, xheight, xheightprev, drecord, drecordprev);
			xheightprev = (int)xheight[4];
			drecordprev = (int)drecord[6];
			
			model.increment();
			curheight = this.model.getS().im();
		}
		RiemannZetaPanel.setDimensions(3840, 2160);
		this.mPanel.zoomlevel();
		RiemannZetaPanel.UNIT = 64;
		try {
			Thread.sleep(5000);
			this.mPanel.saveImage((int)(curheight));
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		this.mPanel.repaint();
		RiemannZetaPanel.setDimensions(WINDOW_WIDTH, WINDOW_HEIGHT);
		RiemannZetaPanel.UNIT = 32;
		
		// Transition to loop
	}
	
	private void loop() {
		int xheightprev = 0;
		int drecordprev = 0;
		long globalTimer, globalTimeElapsed = 0;
		short timeremaining = 0;
		while(true) {
			try {
				globalTimer = System.currentTimeMillis();
				Complex s = model.getS();
				Complex z = model.getZetaS();
				if (model.getOffset() % RiemannZetaCriticalModel.INCREMENT_LEVEL == 0)
					System.out.printf("ζ(%.2f + %.6f\u2148) =\t %.6f %c %.6f\u2148\t(|ζ| = %.6f)\n", s.re(), s.im(), z.re(), z.im() >= 0 ? '+' : '-', Math.abs(z.im()), z.abs());
				
				// Print height
				double[] xheight = model.getXHeight();
				double[] drecord = model.getDrecord();
				printrecord(false, xheight, xheightprev, drecord, drecordprev);
				xheightprev = (int)xheight[4];
				drecordprev = (int)drecord[6];
				
				// Update countdown
				this.mPanel.zoomlevel();
				this.mPanel.countdown();
				this.setTitle(String.format("Million Dollar Limaçon (%d×%d)", this.getBounds().height, this.getBounds().width));
				
				// Increment
				model.increment();
				globalTimeElapsed = System.currentTimeMillis() - globalTimer;
				timeremaining = (short) (SLEEPFRAME - globalTimeElapsed);
				Thread.sleep(timeremaining < 0 ? 0 : timeremaining);
			} catch (Exception exception) {
				exception.printStackTrace();
			}
		}
	}
	
	private static class DrawPanel implements Runnable {
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
	
	public static void main(String[] args) {
		RiemannZetaWindow window = new RiemannZetaWindow();
		if (SKIPTO > 0)
			window.skip((int)(INITIAL + SKIPTO), CHECKPOINT);
		DrawPanel th1 = new DrawPanel(window);
		try {
			for (int k = COUNTDOWN; k > 0; k--) {
				System.out.printf("\rCountdown: %2d", k);
				Thread.sleep(1000);
			}
			System.out.println();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		th1.start();
		window.loop();
	}
}
