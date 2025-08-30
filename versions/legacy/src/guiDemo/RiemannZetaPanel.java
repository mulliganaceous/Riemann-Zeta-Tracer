package guiDemo;

import java.awt.Color;
import java.awt.Graphics;
import java.util.Observable;
import java.util.Observer;
import javax.swing.JPanel;

import complex.Complex;

public class RiemannZetaPanel extends JPanel implements Observer {
	private static final long serialVersionUID = -1;
	
	private RiemannZetaCriticalModel model;
	private final static int LENGTH = 800;
	private final static int HEIGHT = 600;
	public final static int ORIGIN_X = LENGTH/4;
	public final static int ORIGIN_Y = HEIGHT/2;
	public static int UNIT = 32;
	
	public RiemannZetaPanel(RiemannZetaCriticalModel model)  {
		this.model = model;
		
		this.setBackground(Color.BLACK);
		this.setSize(LENGTH,HEIGHT);
		this.setLocation(8,8);
		
		this.model.addObserver(this);
		this.model.setS(new Complex(0.5,0));
	}
	
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		long globalTimer = System.currentTimeMillis();
		System.out.println("Term\tms Elapsed");
		
		this.model.getCommand().executeTraceDemo(g);
		
		g.setColor(new Color(255, 255, 255, 128));
		g.drawLine(0, ORIGIN_Y, LENGTH, ORIGIN_Y);
		g.drawLine(ORIGIN_X, 0, ORIGIN_X, HEIGHT);
		for (int k = 2; k <= 16; k++) {
			g.drawLine(ORIGIN_X+UNIT*k, ORIGIN_Y-4, ORIGIN_X+UNIT*k, ORIGIN_Y+4);
			g.drawLine(ORIGIN_X-UNIT*k, ORIGIN_Y-4, ORIGIN_X-UNIT*k, ORIGIN_Y+4);
			g.drawLine(ORIGIN_X-4, ORIGIN_Y+UNIT*k, ORIGIN_X+4, ORIGIN_Y+UNIT*k);
			g.drawLine(ORIGIN_X-4, ORIGIN_Y-UNIT*k, ORIGIN_X+4, ORIGIN_Y-UNIT*k);
		}
		g.setColor(Color.WHITE);
		g.drawLine(ORIGIN_X, ORIGIN_Y-UNIT/8, ORIGIN_X, ORIGIN_Y+UNIT/8);
		g.drawLine(ORIGIN_X-UNIT/8, ORIGIN_Y, ORIGIN_X+UNIT/8, ORIGIN_Y);
		g.drawOval(ORIGIN_X-UNIT, ORIGIN_Y-UNIT, UNIT*2, UNIT*2);
		
		globalTimer = System.currentTimeMillis() - globalTimer;
		System.out.printf("   T:\t%d:%.3f\n", 
				globalTimer/60000, (globalTimer % 60000)/1000f);
	}

	@Override
	public void update(Observable arg0, Object arg1) {
		this.repaint();
	}
}