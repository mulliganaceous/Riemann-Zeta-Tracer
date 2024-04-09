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
	private int[] form = new int[] {0,0};
	private double[] target = {Double.NaN, Double.NaN};
	// Offset and load
	private int offset = 0;
	private int load = 0;
	private OrbitRiemannCommand command;
	
	public static final int INCREMENT_LEVEL = 1024;
	public static final double INCREMENT = 1./INCREMENT_LEVEL;
	public static int ACCURACY_LEVEL = 2<<16;
	public static final double THRESHOLD = 4;
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
		return this.path[(offset < 0 ? offset + this.path.length : offset) % this.path.length];
	}
	
	public int getZeroes() {
		return this.zeroes;
	}
	
	public int[] getForm() {
		return this.form;
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
	
	private void updateZero(double prevheight) {
		this.dheight = prevheight - this.height;
		this.darc = this.zeropath;
		if (this.dheight < this.xheight[0]) {
			this.xheight[0] = this.dheight;
			this.xheight[2] = this.height;
			this.xheight[4] += 1;
		}
		if (this.zeropath < this.drecord[2]) {
			this.drecord[2] = this.zeropath;
			this.drecord[3] = this.height;
			this.drecord[6] += 2;
		}
		if (this.zeropath > this.drecord[4]) {
			this.drecord[4] = this.zeropath;
			this.drecord[5] = this.height;
			this.drecord[6] += 4;
		}
		this.height = prevheight;
		this.zeropath = 0;
		this.zeroes++;
		this.form[1] = 2;
	}

	private void increment(double dy) {
		double magnitude = this.zetaS.abs();
		int quadrant = this.zetaS.re() < 0 ? 1 : 0;
		quadrant += this.zetaS.im() < 0 ? 2 : 0;
		Complex prev = this.path[(this.offset + this.path.length - 1)%this.path.length];
		Complex dprev = this.dzetaS[0];
		double pathseg = this.dzetaS[0] == null ? 0 : this.dzetaS[0].abs();
		this.path[this.offset] = this.zetaS;
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
			this.speed = pathseg*INCREMENT_LEVEL;
			this.zeropath += pathseg;
			if (this.speed > this.drecord[0]) {
				this.drecord[0] = this.speed;
				this.drecord[1] = this.s.im();
				this.drecord[6] += 1;
			}
			if (this.load > 2) {
				this.dzetaS[1] = this.dzetaS[0].sub(dprev);
			}
		}
		// Zero detection
		if (this.load >= 3) {
			double dmag = this.zetaS.sqnorm() - this.getPath(offset - 2).sqnorm();
			if ((this.form[1] & 2) != 0) {
				this.form[1] = 0;
			}
			if (this.form[1] == 0 && (magnitude <= pathseg*THRESHOLD || magnitude <= 1/1000.) && dmag < 0) {
				this.form[0] = quadrant;
				this.form[1] = 1;
				this.target[1] = magnitude;
				System.out.print(">>> ");
			}
			else if (this.form[1] == 1) {
				double prevheight = this.s.im();
				System.out.print(this.form[0] + "" +  quadrant + " ");
				if (magnitude < this.target[1]) {
					this.target[1] = magnitude;
					this.target[0] = prevheight;
				}
				if ((this.form[0] ^ quadrant) == 3) {
					this.updateZero(prevheight);
					if (magnitude <= pathseg*THRESHOLD)
						System.out.printf("\nZero detected at height %.6f\u2148\007\n", prevheight);
					else {
						System.out.printf("\nZero detected at height %.6f\u2148 and accuracy factor %.6f [I]\n", this.target[0], pathseg*THRESHOLD/this.target[1]);
					}
					this.form[1] = 2;
				}
				else if (magnitude >= pathseg*THRESHOLD && magnitude >= 1/1000.){
					if (this.target[1] < pathseg*THRESHOLD/2) {
						this.updateZero(prevheight);
						System.out.printf("\nZero detected at height %.6f\u2148 and accuracy factor %.6f", this.target[0], pathseg*THRESHOLD/this.target[1]);
						if ((this.form[0] ^ quadrant) == 2) {
							System.out.print(" [V]");
							this.form[1] |= 8;
						}
						if ((this.form[0] ^ quadrant) == 1) {
							System.out.print(" [H]");
							this.form[1] |= 4;
						}
						System.out.print("\007\n");
						this.form[1] |= 2;
					}
					else {
						System.out.printf("\nNo zero around height %.6f, accuracy factor %.6f, \u2148\n", this.target[0], pathseg*THRESHOLD/this.target[1]);
						this.form[1] = -1;
					}
				}
			}
		}
		this.setS(new Complex(s.re(), s.im() + dy));
	}
	
	public void increment() {
		this.increment(1./INCREMENT_LEVEL);
	}
	
	public OrbitRiemannCommand getCommand() {
		return this.command;
	}
	
	public void graph() {
		this.setChanged();
		this.notifyObservers();
	}
}
