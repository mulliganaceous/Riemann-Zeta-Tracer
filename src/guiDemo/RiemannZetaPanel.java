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
	public final static int LENGTH = 640;
	public final static int HEIGHT = 480;
	public final static int ORIGIN_X = LENGTH/4;
	public final static int ORIGIN_Y = HEIGHT/2;
	public static int UNIT = 32;
	
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
		long globalTimer = System.currentTimeMillis();
		this.model.getCommand().executeTraceDemo(g);
		globalTimer = System.currentTimeMillis() - globalTimer;
		System.out.printf("\tRepaint: %d ms\n", globalTimer);
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
			ImageIO.write(bi, "gif", new File("images/" + str + ".gif"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public void update(Observable arg0, Object arg1) {
		this.repaint();
	}
}