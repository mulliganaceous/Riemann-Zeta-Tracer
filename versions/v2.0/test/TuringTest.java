package test;

import static turing.Turing.BACKWARD;
import static turing.Turing.FORWARD;

import turing.Tape;
import turing.Turing;

class TuringTest {
	public static void tapetest() {
		Tape t = new Tape(new char[]{'-', '0', '1'}, "OMG", 4);
		System.out.println(t);
		t.write('1');
		System.out.println(t);
		t.fwd();
		System.out.println(t);
		t.rev();
		System.out.println(t);
		t.rev();
		System.out.println(t);
	}
	
	public static Turing BB4() {
		Turing T = new Turing(5, new char[] {'-','1'});
		T.addTransition(0, '-', 1, '1' ,FORWARD);
		T.addTransition(0, '1', 1, '1' ,BACKWARD);
		T.addTransition(1, '-', 0, '1' ,BACKWARD);
		T.addTransition(1, '1', 2, '-' ,BACKWARD);
		T.addTransition(2, '-', 4, '1' ,FORWARD);
		T.addTransition(2, '1', 3, '1' ,BACKWARD);
		T.addTransition(3, '-', 3, '1' ,FORWARD);
		T.addTransition(3, '1', 0, '-' ,FORWARD);
		return T;
	}
	
	public static void main(String[] args) {
		Turing T = new Turing(4, new char[] {'-','0','1'});
		T.addTransition(0, '0', 1, '-', FORWARD);
		T.addRejection (0, '1');
		T.addAcceptance(0, '-');
		
		T.addTransition(1, '0', 1, '0', FORWARD);
		T.addTransition(1, '1', 1, '1', FORWARD);
		T.addTransition(1, '-', 2, '-', BACKWARD);
		
		T.addRejection (2, '0');
		T.addTransition(2, '1', 3, '-', BACKWARD);
		T.addRejection (2, '-');
		
		T.addTransition(3, '0', 3, '0', BACKWARD);
		T.addTransition(3, '1', 3, '1', BACKWARD);
		T.addTransition(3, '-', 0, '-', FORWARD);
		
		T.initiate("0101", 0);
		T.run(500);
		
		T.initiate("0011", 0);
		T.run(500);
		
		T.initiate("000111", 0);
		T.run(500);
		
		T.initiate("", 0);
		T.run(500);
		
		T = BB4();
		T.run(500);
	}
}
