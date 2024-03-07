package complex;

public abstract class C {
	public abstract double re();
	public abstract double im();
	public abstract double abs();
	public abstract double arg();
	public abstract C conj();
	
	public abstract C add(C other);
	public abstract C sub(C other);
	public abstract C mult(C other);
	public abstract C div(C other);
	
	public abstract C convert();
	public abstract Complex rect();
	public abstract ComplexPolar polar();
}
