package guiDemo;

import javax.swing.BorderFactory;
import javax.swing.JPanel;
import java.awt.*;
import java.util.*;


public class MandelbrotTracePanel extends JPanel implements Observer {
	private MandelbrotTraceModel model;
	final static int ORIGIN = 300;
	final static int UNIT = 128;
	
	public MandelbrotTracePanel(MandelbrotTraceModel model) {
		super();
		this.model = model;
		
		this.setBackground(Color.WHITE);
		this.setSize(600,600);
		this.setLocation(0, 0);
		this.setBorder(BorderFactory.createBevelBorder(1));
	}

	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		
		g.drawLine(ORIGIN-10, ORIGIN, ORIGIN+10, ORIGIN);
		g.drawLine(ORIGIN, ORIGIN-10, ORIGIN, ORIGIN+10);
		g.drawOval(ORIGIN-UNIT, ORIGIN-UNIT, UNIT*2, UNIT*2);
		g.drawOval(ORIGIN-UNIT*2, ORIGIN-UNIT*2, UNIT*4, UNIT*4);
		
		double time = System.currentTimeMillis();
		this.model.getCommand().execute(g);
		this.model.getCommand().executeDots(g);
		System.out.println("Result for " + this.model.getComplex() 
			+ ":\t" + this.model.getResult());
		System.out.println("Operation took " + 
				(System.currentTimeMillis() - time) + "ms");
	}
	
	@Override
	public void update(Observable arg0, Object arg1) {
		this.repaint();
	}
}
