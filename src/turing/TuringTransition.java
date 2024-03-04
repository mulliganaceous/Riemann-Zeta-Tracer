package turing;

public class TuringTransition {
	private final Turing T;
	private final Transition[][] delta;
	public TuringTransition(Turing T, int states, char[] alphabet) {
		this.T = T;
		this.delta = new Transition[states][alphabet.length];
		for (int i = 0; i < states; i++) {
			for (int j = 0; j < alphabet.length; j++) {
				this.addRejection(i, j);
			}
		}
	}

	public static TuringTransition BlankTransition(Turing T, int states, char[] alphabet) {
		return new TuringTransition(T, states, alphabet);
	}
	
	public void addTransition(int state, int symbolIndex, int nextState, int nextSymbolIndex, byte dir) {
		this.delta[state][symbolIndex] = new Transition(nextState, nextSymbolIndex, dir, Turing.RUNNING);
	}
	
	public void addAcceptance(int state, int symbolIndex) {
		this.delta[state][symbolIndex] = new Transition(state, symbolIndex, (byte)0, Turing.APPROVED);
	}
	
	public void addRejection(int state, int symbolIndex) {
		this.delta[state][symbolIndex] = new Transition(state, symbolIndex, (byte)0, Turing.REJECTED);
	}
	
	public Transition getTransition(int state, int symbolIndex) {
		return this.delta[state][symbolIndex];
	}
	
	public class Transition {
		public final int nextState;
		public final int nextSymbolIndex;
		public final byte dir;
		public final byte runningStatus;
		public Transition(int nextState, int nextSymbolIndex, byte dir, byte runningStatus) {
			this.nextState = nextState;
			this.nextSymbolIndex = nextSymbolIndex;
			this.dir = dir;
			this.runningStatus = runningStatus;
		}
		
		public String toString() {
			StringBuilder sb = new StringBuilder();
			if (this.runningStatus == Turing.RUNNING) {
				sb.append("(q" + this.nextState + ",");
				sb.append(TuringTransition.this.T.getAlphabet()[this.nextSymbolIndex] + ")");
				if (this.dir == Turing.FORWARD)
					sb.append(" >>");
				if (this.dir == Turing.BACKWARD)
					sb.append(" <<");
			}
			else if (this.runningStatus == Turing.APPROVED) {
				sb.append("ACCEPT");
			}
			else if (this.runningStatus == Turing.REJECTED) {
				sb.append("REJECT!");
			}
			else
				throw new RuntimeException();
			return sb.toString();
		}
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		Transition transition;
		for (int state = 0; state < this.delta.length; state++) {
			for (int index = 0; index < this.delta[0].length; index++) {
				sb.append("(" + state + "," + this.T.getAlphabet()[index] + ") => ");
				transition = this.delta[state][index];
				sb.append(transition);
				sb.append("\n");
			}
			sb.append("\n");
		}
		return sb.toString();
	}
}
