package guiDemo;

import java.util.Observable;

import complex.Complex;
import riemannzeta.CriticalZeta;

public class RiemannZetaCriticalModel extends Observable {
	private Complex s;
	private Complex zetaS;
	private double interval;
	private Complex[] path;
	// Records
	private int zeroes = 0;
	private double height = Double.NaN;
	private double dheight = Double.NaN;
	private double darc = Double.NaN;
	private double[] xheight = new double[] {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN, Double.NaN, 0};
	private Complex[] dzetaS = new Complex[] { null, null };
	private double speed = -1;
	private double zeropath = 0;
	private double[] drecord = new double[] {Double.NEGATIVE_INFINITY, Double.NaN, Double.POSITIVE_INFINITY, Double.NaN, Double.NEGATIVE_INFINITY, Double.NaN, 0};
	// Zero finding
	private double[] form = new double[] {-1,-1,-1};
	private double target = Double.NaN;
	// Offset and load
	private int offset = 0;
	private int load = 0;
	private OrbitRiemannCommand command;
	
	public static final int INCREMENT_LEVEL = 1024;
	public static final double INCREMENT = (double) 1/INCREMENT_LEVEL;
	public static int ACCURACY_LEVEL = 65536; //1048576
	public static final double THRESHOLD = 16;
	public static final int BUFFERFACTOR = 2;
	
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
	
	public Complex[] getDzetaS() {
		return this.dzetaS;
	}
	
	public double getSpeed() {
		return this.speed;
	}
	
	public double getZeropath() {
		return this.zeropath;
	}
	
	public double getDArc() {
		return this.darc;
	}
	
	public double[] getDrecord() {
		return this.drecord;
	}
	
	public int getOffset() {
		return this.offset;
	}
	
	public int getLoad() {
		return this.load;
	}
	
	public int getMaxLoad() {
		return this.path.length/BUFFERFACTOR;
	}
	
	public void setS(Complex s) {
		this.s = s;
		this.zetaS = CriticalZeta.zeta(s, ACCURACY_LEVEL);
	}
	
	public void setInterval(double b) {
		this.interval = b;
		this.path = new Complex[(int)b*INCREMENT_LEVEL*BUFFERFACTOR];
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
		Complex prev = this.path[(this.offset + this.path.length - 1)%this.path.length];
		Complex dprev = this.dzetaS[0];
		this.path[this.offset] = this.zetaS;
		this.form[this.offset % 3] = magnitude; 
		this.offset++;
		this.offset %= this.path.length;
		if (this.load < this.path.length/BUFFERFACTOR) {
			this.load++;
		}
		// Distance
		this.xheight[4] = 0;
		if (magnitude > this.xheight[1]) {
			this.xheight[1] = magnitude;
			this.xheight[3] = this.s.im();
			this.xheight[4] += 2;
		}
		// Velocity and acceleration
		this.drecord[6] = 0;
		if (this.load > 1) {
			this.dzetaS[0] = this.zetaS.sub(prev);
			this.speed = this.dzetaS[0].abs()*INCREMENT_LEVEL;
			this.zeropath += this.dzetaS[0].abs();
			if (this.speed > this.drecord[0]) {
				this.drecord[0] = this.speed;
				this.drecord[1] = this.s.im();
				this.drecord[6] += 1;
			}
			if (this.load > 2) {
				this.dzetaS[1] = this.dzetaS[0].sub(dprev);
			}
		}
		if (this.load >= 3) {
			if (Double.isNaN(this.target) && (magnitude <= 1./THRESHOLD)) {
				this.target = magnitude;
			}
			else if (!Double.isNaN(this.target)) {
				// System.out.printf("\t%f,^%f, %f\n", this.form[(offset)%3] , this.form[(offset + 1)%3], this.form[(offset + 2)%3]);
				double prevheight = this.s.im() - dy;
				if (this.localminima()) {
					this.target = magnitude;
					this.dheight = prevheight - this.height;
					this.darc = this.zeropath;
					if (this.dheight < this.xheight[0]) {
						this.xheight[0] = this.dheight;
						this.xheight[2] = this.height;
						this.xheight[4] += 1;
					}
					if (this.zeropath < this.drecord[2]) {
						this.drecord[2] = this.zeropath;
						this.drecord[3] = this.s.im();
						this.drecord[6] += 2;
					}
					if (this.zeropath > this.drecord[4]) {
						this.drecord[4] = this.zeropath;
						this.drecord[5] = this.s.im();
						this.drecord[6] += 4;
					}
					this.height = prevheight;
					this.zeropath = 0;
					this.zeroes++;
					System.out.printf("Zero detected at height %.6f\u2148\007\n", prevheight);
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
		// long globalTimer = System.currentTimeMillis();
		this.increment(1/(double) INCREMENT_LEVEL);
		// globalTimer = System.currentTimeMillis() - globalTimer;
		// System.out.printf("\tEvaluate: %d ms\n", globalTimer);
	}
	
	public OrbitRiemannCommand getCommand() {
		return this.command;
	}
	
	public void graph() {
		this.setChanged();
		this.notifyObservers();
	}
}
