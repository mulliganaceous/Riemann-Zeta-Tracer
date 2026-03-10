package guiDemo;

import java.util.Observable;

import complex.Complex;
import riemannzeta.*;

public class RiemannZetaCriticalModel extends Observable {
	private Complex s;
	private Complex zetaS;
	private double interval;
	private OrbitRiemannCommand command;
	
	public static final int INCREMENT_LEVEL = 32;
	public static final double INCREMENT = (double) 1/INCREMENT_LEVEL;
	public static final int ACCURACY_LEVEL = 1048576; //1048576
	
	public RiemannZetaCriticalModel() {
		this.setS(new Complex(0.5,0));
		this.interval = 0;
		this.command = new OrbitRiemannCommand(this);
	}
	
	public Complex getS() {
		return this.s;
	}
	
	public Complex getZetaS() {
		return this.zetaS;
	}
	
	public double getInterval() {
		return this.interval;
	}
	
	public void setS(Complex s) {
		this.s = s;
		this.zetaS = CriticalZeta.zeta(s, ACCURACY_LEVEL);
	}
	
	
	public void setInterval(double b) {
		this.interval = b;
	}
	
	public void increment(double dy) {
		setS(new Complex(s.re(), s.im() + dy));
	}
	
	public void increment() {
		increment(1/(double) INCREMENT_LEVEL);
	}
	
	public OrbitRiemannCommand getCommand() {
		return this.command;
	}
	
	public void graph() {
		this.setChanged();
		this.notifyObservers();
	}
	
	public void run() {
		this.setS(new Complex(0.5, 0));
		while (this.s.im() < 3) {
			this.setInterval(INCREMENT);
			this.graph();
			this.increment();
		}
	}
}
