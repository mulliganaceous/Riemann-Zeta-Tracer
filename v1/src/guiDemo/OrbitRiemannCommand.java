package guiDemo;

import java.awt.Color;
import java.awt.Graphics;

import complex.Complex;
import riemannzeta.CriticalZeta;

public class OrbitRiemannCommand {
	private RiemannZetaCriticalModel model;
	
	public OrbitRiemannCommand(RiemannZetaCriticalModel model) {
		this.model = model;
	}
	
	public void execute(Graphics g) {	
		final int OO = RiemannZetaPanel.ORIGIN_X;
		final int U = RiemannZetaPanel.UNIT;
		int x = (int) (OO + U * this.model.getZetaS().re());
		int y = (int) (OO - U * this.model.getZetaS().im());
		g.drawRect(x - 2, y - 2, 4, 4);
		
		System.out.println(model.getZetaS());
	}
	
	public void executeTraceDemo(Graphics g) {
		final int Ox = RiemannZetaPanel.ORIGIN_X;
		final int Oy = RiemannZetaPanel.ORIGIN_Y;
		final int U = RiemannZetaPanel.UNIT;
		
		int x = (int) (Ox + U * this.model.getZetaS().re());
		int y = (int) (Oy - U * this.model.getZetaS().im());
		int x2, y2;
		Complex s2 = this.model.getS();
		
		long timer = System.currentTimeMillis();
		double b = this.model.getInterval();
		for (double k = 0; k <= b; k += RiemannZetaCriticalModel.INCREMENT) {
			// Timer
			if (k % 1 == 0) {
				System.out.printf("%4d:\t%d\n", 
						(int) k, System.currentTimeMillis() - timer);
				timer = System.currentTimeMillis();
			}
			// Code and graphing runs here
			s2 = s2.add(new Complex(0,RiemannZetaCriticalModel.INCREMENT));
			Complex z2 = CriticalZeta.zeta(s2, 65536);
			x2 = (int) (Ox + U * z2.re());
			y2 = (int) (Oy - U * z2.im());
			Color curColor = Color.getHSBColor((float) (3/(4f*b)*k), 0.75f, 1);
			g.setColor(curColor);
			g.drawLine(x, y, x2, y2);
			
			x = x2;
			y = y2;
		}
	}
	
	public void executeVector(Graphics g) {
		final int Ox = RiemannZetaPanel.ORIGIN_X;
		final int Oy = RiemannZetaPanel.ORIGIN_Y;
		final int U = RiemannZetaPanel.UNIT;
		
		int x = (int) (Ox + U * this.model.getS().re());
		int y = (int) (Oy - U * this.model.getS().im());
		int x2, y2;
		Complex term;
		
		int b = RiemannZetaCriticalModel.ACCURACY_LEVEL;
		b=65536;
		for (int k = 1; k <= b ; k++) {
			term = CriticalZeta.zetaTerm(this.model.getS(), k);
			x2 = (int) (Ox + U * term.re());
			y2 = (int) (Oy - U * term.im());
			
			g.setColor(getColorScheme(k));
			g.drawRect(x2, y2, 2, 2);
			//g.drawLine(x, y, x2, y2);
			
			x = x2;
			y = y2;
		}
	}
	
	public void executeSum(Graphics g) {
		final int Ox = RiemannZetaPanel.ORIGIN_X;
		final int Oy = RiemannZetaPanel.ORIGIN_Y;
		final int U = RiemannZetaPanel.UNIT;
		
		Complex z2 = new Complex(0,0);
		int x = (int) (Ox + U * z2.re());
		int y = (int) (Oy - U * z2.im());
		int x2, y2;
		Complex term;
		
		int b = RiemannZetaCriticalModel.ACCURACY_LEVEL;
		for (int k = 1; k <= b ; k++) {
			term = CriticalZeta.zetaTerm(this.model.getS(), k);
			z2 = z2.add(term);
			x2 = (int) (Ox + U * z2.re());
			y2 = (int) (Oy - U * z2.im());
			
			g.setColor(getColorScheme(k));
			g.drawRect(x2 - 1, y2 - 1, 2, 2);
			//g.drawLine(x, y, x2, y2);
			double log65536 = Math.log(k)/Math.log(2)*32;
			g.drawLine((int)(log65536), 10, (int)(log65536), 18);
			g.drawLine(k/256, 1, k/256, 9);
			
			x = x2;
			y = y2;
		}
		
		x = (int) (Ox + U * this.model.getZetaS().re());
		y = (int) (Oy - U * this.model.getZetaS().im());
		g.setColor(Color.WHITE);
		g.fillRect(x-2, y-2, 4, 4);
	}
	
	private Color getColorScheme(int k) {
		double hue = (5/6f)*(1 - 1/(1-1/2048f+1/2048f*k));
		return Color.getHSBColor((float)hue, 0.75f, 1f);
	}
}
