package guiDemo;

import java.util.Observable;

import complex.Complex;
import riemannzeta.CriticalZeta;

public class RiemannZetaCriticalModel extends Observable {
	private Complex s;
	private Complex zetaS;
	private double interval;
	private Complex[] path;
	private int zeroes = 0;
	private double height = Double.NaN;
	private double dheight = Double.NaN;
	private double[] xheight = new double[] {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN, Double.NaN, 0};
	private double[] form = new double[] {-1,-1,-1};
	private double target = Double.NaN;
	private int offset = 0;
	private int load = 0;
	private OrbitRiemannCommand command;
	
	public static final int INCREMENT_LEVEL = 64;
	public static final double INCREMENT = (double) 1/INCREMENT_LEVEL;
	public static final int ACCURACY_LEVEL = 1048576; //1048576
	public static final double THRESHOLD = 8;
	
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

	public Complex getPath(int offset) {
		if (offset < 0) {
			offset += this.path.length;
		}
		return this.path[offset % this.path.length];
	}
	
	public int getZeroes() {
		return this.zeroes;
	}
	
	public double getHeight() {
		return this.height;
	}
	
	public double getDHeight() {
		return this.dheight;
	}
	
	public double[] getXHeight() {
		return this.xheight;
	}
	
	public int getOffset() {
		return this.offset;
	}
	
	public int getLoad() {
		return this.load;
	}
	
	public int getMaxLoad() {
		return this.path.length;
	}
	
	public void setS(Complex s) {
		this.s = s;
		this.zetaS = CriticalZeta.zeta(s, ACCURACY_LEVEL);
	}
	
	public void setInterval(double b) {
		this.interval = b;
		this.path = new Complex[(int)b*INCREMENT_LEVEL*3];
		this.offset = 0;
		this.load = 0;
		this.zeroes = 0;
	}
	
	public boolean localminima() {
		return (this.load >= 3 && 
				this.form[(offset + 2)%3] > this.form[(offset + 1)%3] && 
				this.form[(offset)%3] > this.form[(offset + 1)%3]);
	}

	private void increment(double dy) {
		double magnitude = this.zetaS.abs();
		this.path[this.offset] = this.zetaS;
		this.form[this.offset % 3] = magnitude; 
		this.offset++;
		this.offset %= this.path.length;
		if (this.load < this.path.length) {
			this.load++;
		}
		this.xheight[4] = 0;
		if (magnitude > this.xheight[1]) {
			this.xheight[1] = magnitude;
			this.xheight[3] = this.s.im();
			this.xheight[4] += 2;
		}
		if (this.load >= 3) {
			if (Double.isNaN(this.target) && (magnitude <= 1./THRESHOLD)) {
				this.target = magnitude;
			}
			else if (!Double.isNaN(this.target)) {
				System.out.printf("\t%f,^%f, %f\n", this.form[(offset)%3] , this.form[(offset + 1)%3], this.form[(offset + 2)%3]);
				if (this.localminima()) {
					this.target = magnitude;
					this.dheight = this.s.im() - this.height;
					if (this.dheight < this.xheight[0]) {
						this.xheight[0] = this.dheight;
						this.xheight[2] = this.height;
						this.xheight[4] += 1;
					}
					this.height = this.s.im();
					this.zeroes++;
					System.out.printf("Zero detected at height %.6f\u2148\007\n", this.s.im() - dy);
				}
				else if (magnitude < this.target) {
					this.target = magnitude;
				}
				else if (magnitude >= 1./THRESHOLD ){
					this.target = Double.NaN;
				}
			}
		}
		
		this.setS(new Complex(s.re(), s.im() + dy));
	}
	
	public void increment() {
		long globalTimer = System.currentTimeMillis();
		this.increment(1/(double) INCREMENT_LEVEL);
		globalTimer = System.currentTimeMillis() - globalTimer;
		System.out.printf("\tEvaluate: %d ms\n", globalTimer);
	}
	
	public OrbitRiemannCommand getCommand() {
		return this.command;
	}
	
	public void graph() {
		this.setChanged();
		this.notifyObservers();
	}
}
