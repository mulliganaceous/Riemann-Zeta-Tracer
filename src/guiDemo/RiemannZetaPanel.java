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
	public final static int LENGTH = 1280;
	public final static int HEIGHT = 720;
	public final static int ORIGIN_X = LENGTH/4;
	public final static int ORIGIN_Y = HEIGHT/2;
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
	
	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		/*
		double sqnorm = this.model.getZetaS().sqnorm();
		if ((this.model.getForm()[1] & 3) != 0) {
			this.countdown = COUNTDOWN;
		}
		if (this.countdown != 0) {
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
			UNIT = 64;
		}
		if (sqnorm < 2) {
			if ((state & 0b100) != 0 && sqnorm < 1E-4) {
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
		this.countdown = (short) (this.countdown == 0 ? 0 : countdown - 1);
		*/
		
		this.model.getCommand().executeTraceDemo(g);
		
		// globalTimer = System.currentTimeMillis() - globalTimer;
		// System.out.printf("\tRepaint: %d ms\n", globalTimer);
	}
	
	public void saveImage(int k) {
	    try {
	    	String str = String.format("Riemann%04d", 
	    			k);
			BufferedImage bi = new BufferedImage
					(LENGTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
			Graphics2D g2 = bi.createGraphics();
			this.repaint();
			this.model.getCommand().executeTraceDemo(g2);
			ImageIO.write(bi, "gif", new File("images/" + str + ".png"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public void update(Observable arg0, Object arg1) {
		this.repaint();
	}
}