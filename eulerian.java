import java.util.ArrayList;
import java.util.Scanner;
public class eulerian {
	static ArrayList<Integer> entered  = new ArrayList<Integer>();
	
	public ArrayList<ArrayList<Integer>> array = new ArrayList<ArrayList<Integer>>();
	
	public static ArrayList<Integer> addelements(int value)
	{
		ArrayList<Integer> enter = new ArrayList<Integer>();
		for (int x = 1;x<=value;x++)
		{
			enter.add(x);
		}
		return enter;
	}
	//constructs all permutations
	public static ArrayList<ArrayList<Integer>> permute(ArrayList<Integer> num)
	{
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		res.add(new ArrayList<Integer>());
		for (int number = 0; number < num.size();number++)
		{
			ArrayList<ArrayList<Integer>> curr = new ArrayList<ArrayList<Integer>>();
			for (ArrayList<Integer> nestedL : res)
			{
				for(int j = 0; j < nestedL.size()+1; j++)
				{
					nestedL.add(j, num.get(number));
					ArrayList<Integer> temp = new ArrayList(nestedL);
					curr.add(temp);
					nestedL.remove(j);
				}
			}
			res = new ArrayList<ArrayList<Integer>>(curr);
		}
		return res;
	}
	
	
	public static int descent(ArrayList<Integer> input)
	{
	
		int output = 0;
		
			for (int y=0; y<input.size()-1;y++)
			{
				if (input.get(y)>input.get(y+1))
				{
					output ++;
				}
			}
		return output;
	}
	//split once function
	public static ArrayList<ArrayList<Integer>> split(ArrayList<Integer> input)
	{
		input.add(100000000);
		ArrayList<Integer> left = new ArrayList<Integer>();
		ArrayList<Integer> middle = new ArrayList<Integer>();
		ArrayList<Integer> right= new ArrayList<Integer>();
		ArrayList<ArrayList<Integer>> value = new ArrayList<ArrayList<Integer>>();
		int smallest = Integer.MAX_VALUE;
		int position = 0;
		if (input.size() == 1)
		{
			if (left.size() !=0)
			{
			value.add(left);
			}
			if (input.size() !=0)
			{
			value.add(input);
			}
			if (right.size() !=0)
			{
			value.add(right);
			}
			return value;
		}
		else
		{
			//find the smallest element and split the array accordingly
			for (int y = 0; y<input.size();y++)
			{
				if (input.get(y)<smallest)
				{
					smallest = input.get(y);
					position = y;
				}
			}
			//create left set
			for (int z = 0; z<position; z++)
			{
				left.add(input.get(z));
			}
			//create middle set
			middle.add(input.get(position));
			//create right set
			for (int a = position+1;a<input.size();a++)
			{
				right.add(input.get(a));
			}
			if (left.size()!=0)
			{
			value.addAll(splitLeft(left));
			}
		if (middle.size() !=0)
		{
		value.add(middle);
		}
		if (right.size() !=0)
		{
		value.add(right);
		}
		return value;
		}
	}
	public static boolean isInOrder(ArrayList<Integer> input)
	{
		
		for (int x = 0; x<input.size()-1;x++)
		{
			if (input.get(x) > input.get(x+1))
			{
				return false;
			}
		}
		return true;
	}
	
	public static ArrayList<ArrayList<Integer>> splitLeft(ArrayList<Integer> input)
	{
		ArrayList<Integer> left = new ArrayList<Integer>();
		ArrayList<Integer> right = new ArrayList<Integer>();
		ArrayList<ArrayList<Integer>> value = new ArrayList<ArrayList<Integer>>();
		int largest = Integer.MIN_VALUE;
		int position = 0;
		if (input.size() == 1 || isInOrder(input))
		{
			value.add(input);
			return value;
		}
		for (int x=0; x<input.size();x++)
		{
			if (input.get(x)>largest)
			{
				largest = input.get(x);
				position = x;
			}
		}
		for (int a = 0; a<=position; a++)
		{
			left.add(input.get(a));
		}
		for (int b = position+1; b<input.size();b++)
		{
			right.add(input.get(b));
		}
		if (left.size() != 0)
		{
		value.add(left);
		}
		input = right;
		
		value.addAll(splitLeft(input));
		
	
	
		return value;
	}
	
	public static int weight(ArrayList<Integer> input)
	{
		int sum = 0;
		//base case
		if (isInOrder (input) || input.size()==1)
		{
			return 0;
		}
		
			for (ArrayList<Integer> z : split(input))
			{
				
				sum = sum + weight(z)+descent(z);
			}
		return sum;
	}
	public static int finalweight(ArrayList<ArrayList<Integer>> input)
	{
		int sum = 0;
		for (ArrayList<Integer> x : input)
		{
			sum = sum+ weight(x) + descent(x);
		}
		return sum;
	}
	public static void printRow(int[] row) {
        for (int i=row.length-1;i>=0;i--) {
            System.out.print(row[i]);
            System.out.print("\t");
        }
        System.out.println();
    }
	
	 public static void main(String[] args) { 
		 String output = "";
		 System.out.println("n=:   (It's advised that n<=10 due to data limitations)");
		 Scanner scan = new Scanner(System.in);
		int ss = scan.nextInt();
		 
		 
		 
		 
		 int sizeofset = ss;
		 
		 
		 
		 
		 
		 
		 
		 int[][] store = new int[15][25];
		 ArrayList<Integer> x = new ArrayList<Integer>();
		 x.add(2);
		 x.add(9);
		 x.add(8);
		 x.add(3);
		 
		 x.add(4);
		 x.add(1);
		 x.add(5);
		 x.add(7);
		 x.add(6);
		 //System.out.println(addelements(5));
		 ArrayList<ArrayList<Integer>> y = permute(addelements(sizeofset));
		 System.out.println(y);
	     //for (ArrayList<Integer> a : y)
	     //{
	    	 //System.out.println(split(a));
	    // }
		 
	   //  System.out.println(split(addelements(4)));
	     try
	     {
	     for (ArrayList<Integer> z : y)
	     {
	    	 	store[descent(z)][finalweight(split(z))] ++;
	    	 	System.out.println(z);
	    	 	System.out.println(descent(z) + "" +(finalweight((split(z)))));
	    	 	
	     }
	     }
	     catch (Exception e)
	     {
	    	 
	     }
	     System.out.println("Coefficients of nth q-Eulerian polynomial (horizontal axis is the power of x, starting from the right, while vertical axis is the power of q, starting from the top):");
	     for(int[] row : store) {
	            printRow(row);
	        }
	     System.out.println("Nathan Sun, nsun1@exeter.edu, 11th grade");
	     System.out.println("Not planning to go abroad. Responsibilities: proctor, Stu Lis, Symphony Orchestra");
	     System.out.println("USACO Gold, CS405, 505, 590, 999. I have some experience in machine learning and deep learning (999 in class). I participate in USACO out of class and I created this program as part of my research at PROMYS to generate data for Eulerian polynomials, something which previously was unknown. See https://arxiv.org/pdf/1702.02446.pdf, https://arxiv.org/pdf/1809.07398.pdf for algorithms");
	     System.out.println("I volunteered and mentored/planned HackExeter and attended USACO training sessions during the fall and winter. I really enjoyed learning how to solve USACO problems, as they are often really clever solutions. The insights I gained from USACO definitely helped me develop as a programmer.");
	     System.out.println("Further, I enjoy applying computer science to other areas of research- the q-Eulerian polynomial project was the best summer of my life.");
	     System.out.println("I would like to help organize ECC events next year and spread interest in CS, which I believe is a vastly underrated topic at Exeter. Also, I would like to organize trips to CS events outside Exeter.");
	     System.out.println("Ideas for ECC next year: perhaps something like HackExeter, but for Exonians. CS is an largely overlooked field in Exeter, and perhaps more could be done to encourage people to start programming.");	    
}
}


