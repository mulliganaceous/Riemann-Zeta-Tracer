package guiDemo;

import static guiDemo.RiemannZetaPanel.HEIGHT;
import static guiDemo.RiemannZetaPanel.LENGTH;
import static guiDemo.RiemannZetaPanel.ORIGIN_X;
import static guiDemo.RiemannZetaPanel.ORIGIN_Y;
import static guiDemo.RiemannZetaPanel.UNIT;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;

import complex.Complex;
import riemannzeta.CriticalZeta;

public class OrbitRiemannCommand {
	private RiemannZetaCriticalModel model;
	private static final float ALPHA = 1f;
	private static final BasicStroke THIN = new BasicStroke(1);
	private static final BasicStroke THICK = new BasicStroke(3);
	public static short countdown[] = {0,0,0,0,0,0};
	public static final short COUNTDOWN = (short)(15);
	
	public OrbitRiemannCommand(RiemannZetaCriticalModel model) {
		this.model = model;
	}
	
	public void grid(Graphics g) {
		// Grid
		g.setColor(new Color(255, 255, 255, 32));
		g.drawLine(0, ORIGIN_Y, LENGTH, ORIGIN_Y);
		g.drawLine(ORIGIN_X, 0, ORIGIN_X, HEIGHT);
		for (int k = 2; k <= 64; k++) {
			g.drawLine(ORIGIN_X+UNIT*k, ORIGIN_Y-4, ORIGIN_X+UNIT*k, ORIGIN_Y+4);
			g.drawLine(ORIGIN_X-UNIT*k, ORIGIN_Y-4, ORIGIN_X-UNIT*k, ORIGIN_Y+4);
			g.drawLine(ORIGIN_X-4, ORIGIN_Y+UNIT*k, ORIGIN_X+4, ORIGIN_Y+UNIT*k);
			g.drawLine(ORIGIN_X-4, ORIGIN_Y-UNIT*k, ORIGIN_X+4, ORIGIN_Y-UNIT*k);
		}
		g.setColor(new Color(255, 255, 255, 128));
		g.drawLine(ORIGIN_X, ORIGIN_Y-UNIT/8, ORIGIN_X, ORIGIN_Y+UNIT/8);
		g.drawLine(ORIGIN_X-UNIT/8, ORIGIN_Y, ORIGIN_X+UNIT/8, ORIGIN_Y);
		g.drawOval(ORIGIN_X-UNIT, ORIGIN_Y-UNIT, UNIT*2, UNIT*2);
		g.drawRect(ORIGIN_X-UNIT/1000, ORIGIN_Y-UNIT/1000, UNIT/500, UNIT/500);
		
		// Top
		String realstr, imstr;
		g.setFont(new Font("Unifont", 0, 16));
		g.drawString(String.format("   s =  0.5      + %.6fi", model.getS().im()), 8, 16);
		
		if (model.getZetaS().re() < 0)
			realstr = String.format(" ζ(s)= -%3.6f", -model.getZetaS().re());
		else
			realstr = String.format(" ζ(s)=  %3.6f", model.getZetaS().re());
		if (model.getZetaS().im() < 0)
			imstr = String.format(" - %3.6fi", -model.getZetaS().im());
		else
			imstr = String.format(" + %3.6fi", model.getZetaS().im());
		g.drawString(realstr + imstr, 8, 32);
		
		Complex dzeta = model.getDzetaS()[0];
		Complex ddzeta = model.getDzetaS()[1];
		if (dzeta != null) {
			if (dzeta.re() < 0)
				realstr = String.format("ζ′(s)= -%3.6f", -dzeta.re()*RiemannZetaCriticalModel.INCREMENT_LEVEL);
			else
				realstr = String.format("ζ′(s)=  %3.6f", dzeta.re()*RiemannZetaCriticalModel.INCREMENT_LEVEL);
			if (dzeta.im() < 0)
				imstr = String.format(" - %3.6fi", -dzeta.im()*RiemannZetaCriticalModel.INCREMENT_LEVEL);
			else
				imstr = String.format(" + %3.6fi", dzeta.im()*RiemannZetaCriticalModel.INCREMENT_LEVEL);
			g.drawString(realstr + imstr, 8, 48);
		}
		if (ddzeta != null) {
			if (ddzeta.re() < 0)
				realstr = String.format("ζ″(s)= -%3.6f", -ddzeta.re()*RiemannZetaCriticalModel.INCREMENT_LEVEL*RiemannZetaCriticalModel.INCREMENT_LEVEL);
			else
				realstr = String.format("ζ″(s)=  %3.6f", ddzeta.re()*RiemannZetaCriticalModel.INCREMENT_LEVEL*RiemannZetaCriticalModel.INCREMENT_LEVEL);
			if (ddzeta.im() < 0)
				imstr = String.format(" - %3.6fi", -ddzeta.im()*RiemannZetaCriticalModel.INCREMENT_LEVEL*RiemannZetaCriticalModel.INCREMENT_LEVEL);
			else
				imstr = String.format(" + %3.6fi", ddzeta.im()*RiemannZetaCriticalModel.INCREMENT_LEVEL*RiemannZetaCriticalModel.INCREMENT_LEVEL);
			g.drawString(realstr + imstr, 8, 64);
			
			Complex ndzeta = dzeta.div(dzeta.abs());
			Complex nddzeta = dzeta.sub(ddzeta);
			nddzeta = nddzeta.div(nddzeta.abs());
			double cross = ndzeta.re()*nddzeta.im() - ndzeta.im()*nddzeta.re();
			g.drawString(String.format("dArg = %+3.3f°", Math.asin(cross)*180/Math.PI*RiemannZetaCriticalModel.INCREMENT_LEVEL), 8, 80);
		}
		
		// Bottom
		if (countdown[0] > 0) {
			g.setColor(countdown[2] != 0 ? (countdown[2] < 0 ? Color.red : Color.yellow) : Color.green);
			countdown[0]--;
		}
		else if (model.getForm()[1] == 1) {
			g.setColor(Color.orange);
		}
		String heightstr = "";
		double height = model.getHeight();
		if (!Double.isNaN(height)) {
			heightstr = String.format("last height: ~%3.3f", height);
		}
		g.drawString(String.format("| ζ(s)|= %-31.3f\t" + heightstr, model.getZetaS().abs()), 8, RiemannZetaPanel.HEIGHT-48);
		
		heightstr = "";
		double zeropathlength = model.getZeropath();
		heightstr = String.format("arc length : %3.3f", zeropathlength);
		g.drawString(String.format("|ζ′(s)|= %-31.3f" + heightstr, model.getSpeed()), 8, RiemannZetaPanel.HEIGHT-32);
		
		heightstr = "";
		double dheight = model.getDHeight();
		double darc = model.getDArc();
		if (!Double.isNaN(dheight)) {
			heightstr = String.format("distance   : ~%3.3f (arc: %3.3f)", dheight, darc);
		}
		g.drawString(String.format(" zeros \u2248 %-31d" + heightstr, model.getZeroes()), 8, RiemannZetaPanel.HEIGHT-16);
		
		// Records
		heightstr = "";
		double[] xheight = model.getXHeight();
		int xind = (int) xheight[4];
		if (!Double.isNaN(xheight[2])) {
			heightstr = String.format("Closest : Δ \u2248 %3.3E @ h1\u2248 %3.3f", xheight[0], xheight[2]);
		}
		g.setColor(new Color(255, 255, 255, 128));
		if (countdown[1] > 0) {
			g.setColor(Color.green);
			countdown[1]--;
		}
		g.drawString(String.format(heightstr, model.getZeroes()), 328, 16);
		
		heightstr = "";
		if (!Double.isNaN(xheight[3])) {
			heightstr = String.format("Farthest:|ζ|\u2248 %3.3f @ h \u2248 %3.3f", xheight[1], xheight[3]);
		}
		g.setColor(new Color(255, 255, 255, 128));
		if ((xind & 2) != 0) {
			g.setColor(Color.green);
		}
		g.drawString(String.format(heightstr, model.getZeroes()), 328, 32);
		
		g.setColor(new Color(255, 255, 255, 128));
		
		double[] drecord = model.getDrecord();
		int dind = (int) drecord[6];
		heightstr = "";
		if (!Double.isNaN(drecord[3])) {
			heightstr = String.format("Shortest: L \u2248 %3.3E @ h \u2248 %3.3f", drecord[2], drecord[3]);
		}
		g.setColor(new Color(255, 255, 255, 128));
		if (countdown[3] > 0) {
			g.setColor(Color.green);
			countdown[3]--;
		}
		g.drawString(String.format(heightstr), 328, 48);
		
		g.setColor(new Color(255, 255, 255, 128));
		
		heightstr = "";
		if (!Double.isNaN(drecord[5])) {
			heightstr = String.format("Longest : L \u2248 %3.3f @ h \u2248 %3.3f", drecord[4], drecord[5]);
		}
		g.setColor(new Color(255, 255, 255, 128));
		if (countdown[4] > 0) {
			g.setColor(Color.green);
			countdown[4]--;
		}
		g.drawString(String.format(heightstr), 328, 64);
		
		g.setColor(new Color(255, 255, 255, 128));
		
		heightstr = "";
		if (!Double.isNaN(drecord[1])) {
			heightstr = String.format("Fastest : L \u2248 %3.3f @ h \u2248 %3.3f", drecord[0], drecord[1]);
		}
		g.setColor(new Color(255, 255, 255, 128));
		if ((dind & 1) != 0) {
			g.setColor(Color.green);
		}
		g.drawString(String.format(heightstr), 328, 80);
		
		g.setColor(new Color(0xFB, 0x22, 0xD0));
		g.fillRect(0, RiemannZetaPanel.HEIGHT-15, 8*61, 16);
		g.setColor(new Color(0x5E, 0x00, 0x45));
		g.drawString("Riemann Zeta Tracer v2.A.FFFF | ⓒ 2018-2024 mulliganaceous", 8, RiemannZetaPanel.HEIGHT-0);
		
		g.setColor(new Color(255, 255, 255, 128));
	}
	
	public void execute(Graphics g) {	
		grid(g);
		final int OO = RiemannZetaPanel.ORIGIN_X;
		final int U = RiemannZetaPanel.UNIT;
		int x = (int) (OO + U * this.model.getZetaS().re());
		int y = (int) (OO - U * this.model.getZetaS().im());
		g.drawRect(x - 2, y - 2, 4, 4);
		
		System.out.println(model.getZetaS());
	}
	
	public void executeTrace(Graphics g) {
		final int Ox = RiemannZetaPanel.ORIGIN_X;
		final int Oy = RiemannZetaPanel.ORIGIN_Y;
		final int U = RiemannZetaPanel.UNIT;
		
		int offset = this.model.getOffset();
		final int load = this.model.getLoad();
		
		Complex z2 = this.model.getPath(-load + offset);
		if (z2 == null) {
			z2 = this.model.getZetaS();
			int x = (int) (Ox + U * z2.re());
			int y = (int) (Oy - U * z2.im());
			g.setColor(Color.WHITE);
			g.drawLine(x - 4, y - 4, x + 4, y + 4);
			g.drawLine(x - 4, y + 4, x + 4, y - 4);
			((Graphics2D) g).setStroke(THIN);
			grid(g);
			return;
		}
		int x = (int) (Ox + U * z2.re());
		int y = (int) (Oy - U * z2.im());
		int x2, y2;
		
		for (int k = 1; k <= load; k++) {
			// Code and graphing runs here
			z2 = k == load ? this.model.getZetaS() : this.model.getPath(-load + offset + k);
			x2 = (int) (Ox + U * z2.re());
			y2 = (int) (Oy - U * z2.im());
			Color curColor = Color.getHSBColor((float) ((13/16f)*(k/(float) model.getMaxLoad())), 0.75f, 1);
			curColor = new Color(curColor.getRed()/255.0f, curColor.getGreen()/255.0f, curColor.getBlue()/255.0f, ALPHA);
			// Process for graphing
	        ((Graphics2D) g).setStroke(THICK);
			g.setColor(curColor);
			g.drawLine(x, y, x2, y2);
			// Update coordinate
			x = x2;
			y = y2;
		}
		g.setColor(Color.WHITE);
		g.drawLine(x - 4, y - 4, x + 4, y + 4);
		g.drawLine(x - 4, y + 4, x + 4, y - 4);
		((Graphics2D) g).setStroke(THIN);
		grid(g);
	}
	
	public void executeVector(Graphics g) {
		grid(g);
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
			g.drawLine(x, y, x2, y2);
			
			x = x2;
			y = y2;
		}
	}
	
	public void executeSum(Graphics g) {
		grid(g);
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
