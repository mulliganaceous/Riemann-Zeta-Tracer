package guiDemo;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Observable;
import java.util.Observer;

import javax.imageio.ImageIO;
import javax.swing.JPanel;

import complex.Complex;

public class RiemannZetaPanel extends JPanel implements Observer {
	private static final long serialVersionUID = -1;
	
	private RiemannZetaCriticalModel model;
	public static int LENGTH = RiemannZetaWindow.WINDOW_WIDTH;
	public static int HEIGHT = RiemannZetaWindow.WINDOW_HEIGHT;
	public static int ORIGIN_X = LENGTH/4;
	public static int ORIGIN_Y = HEIGHT/2;
	public static int UNIT = 32;
	private int state = 0;
	private double arc = 0;
	private static short countdown = 0;
	private static final short COUNTDOWN = (short) (RiemannZetaCriticalModel.INCREMENT_LEVEL/5);
	
	public RiemannZetaPanel(RiemannZetaCriticalModel model)  {
		this.model = model;
		
		this.setBackground(Color.BLACK);
		this.setSize(LENGTH,HEIGHT);
		this.setMinimumSize(new Dimension(LENGTH,HEIGHT));
		this.setLocation(0,0);
		
		this.model.addObserver(this);
		this.model.setS(new Complex(0.5,0));
	}
	
	public void zoomlevel() {
		double sqnorm = this.model.getZetaS().sqnorm();
		if ((this.model.getForm()[1] & 3) != 0) {
			RiemannZetaPanel.countdown = COUNTDOWN;
		}
		if (RiemannZetaPanel.countdown != 0) {
			double arcz = this.model.getDArc();
			if (arcz != arc) {
				state &= ~(3 << 1);
				if (arcz < 0.01) {
					state |= (3 << 1);
				}
				else if (arcz < Math.PI){
					state |= (1 << 1);
				}
				arc = arcz;
			}
		}
		if (sqnorm >= 100) {
			state |= 1;
		}
		else {
			state &= ~1;
		}
		
		if ((state & 1) == 1) {
			UNIT = 16;
		}
		else if (sqnorm < 2) {
			if ((state & 0b100) != 0 && sqnorm < 1/16.) {
				UNIT = 65536;
			}
			else if ((state & 0b110) != 0) {
				UNIT = 256;
			}
		}
		else if ((state & 0b110) != 0 && sqnorm >= 2) {
			state &= ~(3 << 1);
			UNIT = 32;
		}
		else {
			UNIT = 32;
		}
		RiemannZetaPanel.countdown = (short) (RiemannZetaPanel.countdown == 0 ? 0 : countdown - 1);
	}
	
	public static void setDimensions(int length, int height) {
		LENGTH = length;
		HEIGHT = height;
		ORIGIN_X = LENGTH/4;
		ORIGIN_Y = HEIGHT/2;
	}
	
	public void countdown() {
		double[] xheight = model.getXHeight();
		int xind = (int) xheight[4];

		if ((model.getForm()[1] & 2) != 0) {
			OrbitRiemannCommand.countdown[0] = OrbitRiemannCommand.COUNTDOWN;
			OrbitRiemannCommand.countdown[2] = (short) (model.getForm()[1] >> 2);
		}
		if ((xind & 1) == 1) {
			OrbitRiemannCommand.countdown[1] = OrbitRiemannCommand.COUNTDOWN;
		}
		double[] drecord = model.getDrecord();
		int dind = (int) drecord[6];
		if ((dind & 2) != 0) {
			OrbitRiemannCommand.countdown[3] = OrbitRiemannCommand.COUNTDOWN;
		}
		if ((dind & 4) != 0) {
			OrbitRiemannCommand.countdown[4] = OrbitRiemannCommand.COUNTDOWN;
		}
	}
	
	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		RiemannZetaPanel.setDimensions(getWidth(), getHeight());
		this.model.getCommand().executeTrace(g);
		
		// globalTimer = System.currentTimeMillis() - globalTimer;
		// System.out.printf("\tRepaint: %d ms\n", globalTimer);
	}
	
	public void saveImage(int k) {
	    try {
	    	String str = String.format("Riemann0x%04x", k);
			BufferedImage bi = new BufferedImage(LENGTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
			Graphics2D g2 = bi.createGraphics();
			this.model.getCommand().executeTrace(g2);
			ImageIO.write(bi, "png", new File("images/" + str + ".png"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public void update(Observable arg0, Object arg1) {
		this.repaint();
	}
}