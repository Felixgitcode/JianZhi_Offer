# JianZhi_Offer

# 一、前言
本文主要记录个人刷剑指offer的过程，一来加深记忆，二来便于复习加强，题目部分为自己解答，其他的有些参考借鉴了牛客还有其他博客上的答案，望各位大佬轻喷，侵删。
觉得对你有帮助的话，欢迎给个star噢，笔芯！！！！
# 二、题目
#### 1、二维数组中的查找
题目描述：
> 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

解题思路：
从题目给的条件来看：每一行从左到右递增，每一列从上到下递增，把每一行看成有序数组，这正好符合二分查找的规律
```javascript
//二分查找
 public boolean find2(int target, int[][] array) {
   int row = array.length;
   int col = array[0].length;
   if (row == 0 || col == 0)
     return false;
   if (target < array[0][0] || target > array[row - 1][col - 1])
     return false;

   for(int i=0; i<row; i++){
     int low=0,high=col-1,mid;
     while(low <= high){
       mid = low+(high - low)/2;
       if(target == array[i][mid])
         return true;
       else if(target < array[i][mid])
         high = mid -1;
       else
         low = mid +1;
     }
   }
   return false;
 }
```
这道题还有一种思路，利用每一行从左到右递增，每一列从上到下递增的规律，选取右上角或者左下角的元素a与target进行比较，（本题我们选取左下角的点为例）
当target小于元素a时，那么target必定在元素a所在列的上方,此时让元素往上走；
当target大于元素a时，那么target必定在元素a所在行的右边,此时让元素往右走；
```javascript
public boolean find1(int target, int[][] array) {
   int row = array.length;
   int col = array[0].length;
   if (row == 0 || col == 0)
     return false;
   if (target < array[0][0] || target > array[row - 1][col - 1])
     return false;

   int x = row - 1, y = 0;
   while (x >= 0 && y <= col - 1) {//从左下开始判定，遇小向右，遇大向上。
     if (target == array[x][y]){
       return true;
     }else if (target > array[x][y]){
       y++;
     }else{
       x--;
     }
   }

   return false;
 }
```

#### 2、替换空格
题目描述：

> 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are
> Happy.则经过替换之后的字符串为We%20Are%20Happy。

解题思路：
利用辅助空间，创建一个字符数组用来存放字符串，然后利用for循环遍历一遍，遇到空格直接接上“%20”。
```javascript
public String replaceSpace(StringBuffer str) {
	String s = str.toString();
	char[] c = s.toCharArray();
    	StringBuilder sb = new StringBuilder();
    	
    	for(int i=0; i<c.length; i++){
    		if(c[i] == ' '){
    			sb.append("%20");
    		}else{
    			sb.append(c[i]);
    		}
    	}
		
		return sb.toString();
  }
```
#### 3、从头到尾打印链表
题目描述：

> 输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

解题思路：
使用栈，利用栈先进后出的特点将链表从尾到头打印
```javascript
public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
		Stack<Integer> stack = new Stack<Integer>();
		ArrayList<Integer> array = new ArrayList<Integer>();
		
		while(listNode != null){
			stack.push(listNode.val);
			listNode = listNode.next;
		}
		while(!stack.isEmpty()){
			array.add(stack.pop());
		}
		
		return array;
	}
```
使用递归
```javascript
public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
    ArrayList<Integer> array = new ArrayList<>();
    if (listNode != null) {
        array.addAll(printListFromTailToHead(listNode.next));
        array.add(listNode.val);
    }
    return array;
}
```

#### 4、重建二叉树
题目描述：

> 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

解题思路：
二叉树的前序遍历顺序是：先访问根节点，然后遍历左子树，再遍历右子树。 中序遍历顺序是：中序遍历根节点的左子树，然后是访问根节点，最后中序遍历右子树。
1、二叉树的前序遍历序列一定是该树的根节点
2、中序遍历序列中根节点前面一定是该树的左子树，后面是该树的右子树
从上面可知，题目中前序遍历的第一个节点{1}一定是这棵二叉树的根节点，根据中序遍历序列，可以发现中序遍历序列中节点{1}之前的{4,7,2}是这棵二叉树的左子树，
{5,3,8,6}是这棵二叉树的右子树。然后，对于左子树，递归地把前序子序列{2,4,7}和中序子序列{4,7,2}看成新的前序遍历和中序遍历序列。此时，对于这两个序列，
该子树的根节点是{2}，该子树的左子树为{4,7}、右子树为空，如此递归下去（即把当前子树当做树，又根据上述步骤分析）。{5,3,8,6}这棵右子树的分析也是这样。
```javascript
public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
		if(pre.length == 0 || in.length == 0)
			return null;
		
		return reConstructBinaryTree(pre, in, 0, pre.length-1, 0, in.length-1);
	}
	//构建方法，pStart和pEnd分别是前序遍历序列数组的第一个元素和最后一个元素；
	//iStart和iEnd分别是中序遍历序列数组的第一个元素和最后一个元素。
	public TreeNode reConstructBinaryTree(int[] pre,int[] in,int pStart,int pEnd,int iStart,int iEnd){
		TreeNode tree = new TreeNode(pre[pStart]);//建立根结点
		tree.left = null;
		tree.right = null;
		if(pStart == pEnd && iStart == iEnd){//递归终止条件
			return tree;
		}
		int root = 0;
		for(root=iStart; root<iEnd; root++){//遍历中序数组中的根结点
			if(pre[pStart] == in[root]){
				break;
			}
		}
		//划分左右子树
		int leftLength = root - iStart;
		int rightLength = iEnd - root;
		if(leftLength > 0){//重建左子树
			tree.left = reConstructBinaryTree(pre, in, pStart+1, pStart+leftLength, iStart, root-1);
		}
		if(rightLength > 0){//重建右子树
			tree.right = reConstructBinaryTree(pre, in, pStart+leftLength+1, pEnd, root+1, iEnd);
		}
		return tree;
	}
```
#### 5、用两个栈实现队列
题目描述：

> 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

解题思路：
两个栈1和2，栈1用来实现队列的push操作，栈2用来实现pop操作
- 入队：将元素进栈1
- 出队：判断栈2是否为空，如果为空，则将栈1中所有元素pop，并push进栈2，然后栈2出栈；
```javascript
    Stack<Integer> stackPush = new Stack<Integer>();//用于push的栈
    Stack<Integer> stackPop = new Stack<Integer>();//用于pop的栈
    
    public void push(int node) {
        stackPush.push(node);
    }
    
    public int pop() {
    	if(stackPop.isEmpty() && stackPush.isEmpty()){
    		throw new IllegalArgumentException("队列已空！");
    	}else if(stackPop.isEmpty()){
    		while(!stackPush.isEmpty()){
    			stackPop.push(stackPush.pop());
    		}
    	}
    	return stackPop.pop(); 
    }
```

#### 6、旋转数组的最小数字
题目描述：

> 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

解题思路：
采用二分法解答这个问题，mid = low + (high - low)/2，需要考虑三种情况：
- (1)array[mid] > array[high]:
		出现这种情况的array类似[3,4,5,6,0,1,2]，此时最小数字一定在mid的右边。故 low = mid + 1
- (2)array[mid] == array[high]:
		出现这种情况的array类似 [1,0,1,1,1] 或者[1,1,1,0,1]，此时最小数字不好判断在mid左边还是右边,这时只好一个一个试 ，故 high = high - 1
- (3)array[mid] < array[high]:
		出现这种情况的array类似[2,2,3,4,5,6,6],此时最小数字一定就是array[mid]或者在mid的左边。因为右边必然都是递增的。故 high = mid
		注意这里有个坑：如果待查询的范围最后只剩两个数，那么mid 一定会指向下标靠前的数字，比如 array = [4,6] array[low] = 4 ;array[mid] = 4 ; array[high] = 6 ;如果high = mid - 1，就会产生错误， 因此high = mid，但情形(1)中low = mid + 1就不会错误
```javascript
public static int minNumberInRotateArray(int[] array){
		if(array == null || array.length == 0)
			return 0;
		int low=0, high=array.length-1;
		while(low < high){
			int mid = low + (high-low)/2;
			if(array[mid] > array[high]){
				low = mid + 1;
			}else if(array[mid] == array[high]){
				high = high - 1;
			}else{
				high = mid;
			}
		}
		
		return array[low];
	}
```
#### 7、斐波那契数列
题目描述：

> 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。n<=39

解题思路：
在数学上，斐波纳契数列以如下被以递推的方法定义：F(1)=1，F(2)=1, F(3)=2, F(n)=F(n-1)+F(n-2)（n>=4，n∈N*）  
递归法
```javascript
public static int fibonacci(int n){
		if(n <= 0)
			return 0;
		if(n == 1)
			return 1;
		return fibonacci(n-1) + fibonacci(n-2);
	}
```
循环迭代法
```javascript
public static int fibonacci(int n){
		if(n < 2)
			return n;
		int f=0,f1=0,f2=1;
		for(int i=2; i<=n; i++){
			f = f1 + f2;
			f1 = f2;
			f2 = f;
		}
		return f;
	}
```
#### 8、跳台阶
题目描述：

> 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

解题思路：
- （1）第0层，跳法f(0)=0,从0跳到0就是不用跳
-   （2）第1层，跳法f(1)=1，只有是从0层跳一级到达第1层这一种跳法
- （3）当小青蛙位于第n（n>=2）层的时候，它可能是在n-1层跳一步到达的，也可能是在n-2层跳2步到达的，小青蛙跳n层的有f(n)种跳法，n-1层时是f(n-1)种跳法，n-2层时是f(n-2)种跳法，所以f(n)=f(n-1)+f(n-2)
```javascript
    //循环
	public static int jumpFloor(int target){
		if(target < 2)
			return target;
		int res=0, f1=1, f2=1;
		for(int i=1; i<target; i++){
			res = f1+f2;
			f1 = f2;
			f2 = res;
		}
		return res;
	}
	//递归
	public static int jumpFloor2(int target){
		if(target < 0)
			return 0;
		int[] jump = {0,1,2};//1表示一层台阶一种跳法，2表示两种跳法
		if(target < 3)
			return jump[target];
		return jumpFloor(target-1)+jumpFloor(target-2);
	}
```
#### 9、变态跳台阶
题目描述：

> 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

解题思路：
- f(0)=0,f(1) = 1
- 第2层会有两种跳的方式，一次1阶或者2阶，f(2) = f(1) + f(0)
- 第3层 会有三种跳的方式，1阶、2阶、3阶，f(3) = f(2)+f(1)+f(0)
- 第n层会有n种跳的方式，1阶、2阶...n阶，f(n) =  f(0) + f(1) + f(2) + f(3) + ... + f(n-1) 
将f(n)-f(n-1)得到f(n) = 2*f(n-1)，得出结论：f(n) = 2*f(n-1)
```javascript
//递归
	public int JumpFloorII(int target) {
		if (target <= 0)
			return 0;
		if (target == 1)
			return 1;
		return 2 * JumpFloorII(target - 1);
	}
	//循环
	public int JumpFloorII2(int target) {
		if (target <= 0)
			return 0;
		if (target == 1)
			return 1;
		int a = 1;
		int b = 2;
		for (int i = 2; i <= target; i++) {
			b = 2 * a;
			a = b;
		}
		return b;
	}
```
#### 10、 矩形覆盖
题目描述：

> 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

解题思路：
```javascript
//递归
	public int rectCover(int target){
		if(target < 3)
			return target;
		return rectCover(target-1)+rectCover(target-2);
	}
	//循环
	public int rectCover2(int target){
		if(target < 3)
			return target;
		int first=1,second=2,res=0;
		for(int i=3; i<=target; i++){
			res = first+second;
			first = second;
			second = res;
		}
		return res;
	}
```
#### 11、二进制中1的个数
题目描述：

> 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

解题思路：
法1，先把整数转换为二进制字符串,把字符串转为char数组，遍历
```javascript
public static int numberOf1(int n){
		int res=0;
		char[] array = Integer.toBinaryString(n).toCharArray();
		for(int i=0; i<array.length; i++){
			if(array[i] == '1'){
				res++;
			}
		}
		return res;
	}
```
法二， 利用&运算的特性,把一个整数减去1，再和原整数做与运算，会把该整数最右边一个1变成0.那么一个整数的二进制有多少个1，就可以进行多少次这样的操作 
```javascript
public static int numberOf1_3(int n) {
		int res = 0;
		while(n != 0){
			n = n & (n-1);
			res++;
		}
		return res;
	}
```
#### 12、数值的整数次方
题目描述：

> 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

解题思路：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190114234423178.png)

```javascript
public double Power(double base, int exponent) {
    if (exponent == 0)
        return 1;
    if (exponent == 1)
        return base;
    boolean isNegative = false;
    if (exponent < 0) {
        exponent = -exponent;
        isNegative = true;
    }
    double pow = Power(base * base, exponent / 2);
    if ((exponent&1) == 1)//奇数
        pow = pow * base;
    return isNegative ? (1/pow) : pow;
}
```
#### 13、调整数组顺序使奇数位于偶数前面
题目描述：

> 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

解题思路：
创建一个辅助数组，一次遍历计算奇数的数量count，再次遍历将奇数的值依次放在辅助数组的0到count-1位置上，偶数值依次放在count及之后的位置上，最后将辅助数组写回原数组
```javascript
//奇数&1等于1，偶数&1等于0
	public static void reOrderArray(int[] array) {
		if (array.length < 2)
			return;
		int count = 0, begin = 0;
		int[] newArray = new int[array.length];//辅助数组
		for (int i = 0; i < array.length; i++) {
			if ((array[i] & 1) == 1)//计算所有奇数的数量
				count++;
		}
		for (int i = 0; i < array.length; i++) {
			if ((array[i] & 1) == 1)
				newArray[begin++] = array[i];//0到count-1位置放奇数
			else
				newArray[count++] = array[i];//count及之后的位置放偶数
		}
		for (int i = 0; i < array.length; i++) {
			array[i] = newArray[i];//将数据放回原数组
		}
	}
```
#### 14、链表中倒数第k个结点
题目描述：

> 输入一个链表，输出该链表中倒数第k个结点。

解题思路：
```javascript
public ListNode findKthToTail(ListNode head,int k) {
		if(head == null || k<=0)
			return null;
		ListNode cur = head;
		int len = 0;
		while(head != null){//计算链表长度
			++len;
			head = head.next;
		}
		if(len < k)
			return null;
		for(int i=0; i<len-k; i++){//第len-k个就是倒数第k个
			cur = cur.next;
		}
		return cur;
    }
```
#### 15、反转链表
题目描述：

> 输入一个链表，反转链表后，输出新链表的表头。

解题思路：
```javascript
    //递归
	public ListNode reverseList(ListNode head) {
		if(head == null || head.next == null)
			return head;
		ListNode headNode = reverseList(head.next);
		head.next.next = head;
		head.next = null;
		return headNode;
    }
    //循环
	public ListNode reverseList(ListNode head) {
		if (head == null || head.next == null)
			return null;
		
		ListNode pre = null;
		ListNode next = null;
		
		while (head != null) {
			next = head.next;
			head.next = pre;
			pre = head;
			head = next;
		}
		return pre;
	}
```
#### 16、合并两个排序的链表
题目描述：

> 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

解题思路：
```javascript
     //迭代
     public ListNode merge(ListNode list1, ListNode list2) {
		if(list1 == null || list2 == null)
			return list1 == null ? list2 : list1;
		ListNode dummy = new ListNode(0);
		ListNode cur = dummy;
		while(list1 != null && list2 != null){
			if(list1.val < list2.val){
				cur.next = list1;
				list1 = list1.next;
			}else{
				cur.next = list2;
				list2 = list2.next;
			}
			cur = cur.next;
		}
		cur.next = (list1 == null) ? list2 : list1;//当前指针指向剩下的不为空的链表
		return dummy.next;		
	}
	//递归版本
	public ListNode merge(ListNode list1, ListNode list2) {
		if (list1 == null) {
			return list2;
		}
		if (list2 == null) {
			return list1;
		}
		if (list1.val <= list2.val) {
			list1.next = merge(list1.next, list2);
			return list1;
		} else {
			list2.next = merge(list1, list2.next);
			return list2;
		}
	}
```
#### 17、树的子结构
题目描述：

> 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

解题思路：
```javascript
    public boolean hasSubtree(TreeNode root1, TreeNode root2) {
		boolean res = false;
		if(root1 != null && root2 != null){//当Tree1和Tree2都不为零的时候，才进行比较。否则直接返回false
			if(root1.val == root2.val){//如果找到了对应Tree2的根节点的点
				res = doesTree1HaveTree2(root1,root2);//以这个根节点为为起点判断是否包含Tree2
			}
			if(!res){//如果找不到，那么就再去root的左儿子当作起点，去判断是否包含Tree2
				res = hasSubtree(root1.left, root2);
			}
			if (!res) {// 如果还找不到，那么就再去root的右儿子当作起点，去判断是否包含Tree2
				res = hasSubtree(root1.right, root2);
			}
		}
		return res;
	}
	public static boolean doesTree1HaveTree2(TreeNode root1,TreeNode root2){
		//如果Tree2已经遍历完了都能对应的上，返回true
		if(root2 == null)
			return true;
		//如果Tree2还没有遍历完，Tree1却遍历完了。返回false
		if(root1 == null)
			return false;
		//如果其中有一个点没有对应上，返回false
		if(root1.val != root2.val)
			return false;
		//如果根节点对应的上，那么就分别去子节点里面匹配
		return doesTree1HaveTree2(root1.left,root2.left) && doesTree1HaveTree2(root1.right,root2.right) ;
	}
```
#### 18、二叉树的镜像
题目描述：

> 操作给定的二叉树，将其变换为源二叉树的镜像。

解题思路：
递归思想，先交换根结点的左右子树，然后向下递归，分别把左右子树作为下次循环的根结点
```javascript
public void mirror(TreeNode root) {
		if(root == null)
			return;
		if(root.left==null && root.right==null)
			return;
		//交换左右子树
		TreeNode temp = root.left;
		root.left = root.right;
		root.right = temp;
		//循环递归
		if(root.left!=null)
			mirror(root.left);
		if(root.right!=null)
			mirror(root.right);
	}
```
#### 19、顺时针打印矩阵
题目描述：

> 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190115215206293.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZlbGl4X2Fy,size_16,color_FFFFFF,t_70)
解题思路：
```javascript
   public static ArrayList<Integer> printMatrix(int[][] matrix) {
		int Lr = 0,Lc = 0;//左上角点的行和列
		int Rr = matrix.length-1,Rc=matrix[0].length-1;//右下角点的行和列
		ArrayList<Integer> matrixList = new ArrayList<>();
		while(Lr<=Rr && Lc<=Rc){
			//每次打印一圈，相应变换左上角点和邮下角点
			printEdge(matrixList,matrix,Lr++,Lc++,Rr--,Rc--);
		}
		return matrixList;
    }
	
	public static void printEdge(ArrayList<Integer> list,int[][] m,int Lr,int Lc,int Rr,int Rc){
		if(Lr==Rr){//同一行
			for(int i=Lc; i<=Rc; i++){
				list.add(m[Lr][i]);
			}
		}else if(Lc == Rc){//同一列
			for(int i=Lr; i<=Rr; i++){
				list.add(m[i][Lc]);
			}
		}else{
			int curC = Lc;
			int curR = Lr;
			while(curC != Rc){//从左到右打印一行
				list.add(m[Lr][curC]);
				curC++;
			}
			while(curR != Rr){//从上到下打印一列
				list.add(m[curR][Rc]);
				curR++;
			}
			while(curC != Lc){//从右到左打印一行
				list.add(m[Rr][curC]);
				curC--;
			}
			while(curR != Lr){//从下到上打印一行
				list.add(m[curR][Lc]);
				curR--;
			}
		}
	}
```
#### 20、包含min函数的栈
题目描述：

> 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

解题思路：
```javascript
  public class Solution {
    Stack<Integer> dataStack = new Stack<>();//数据栈，存放数据
	Stack<Integer> minStack = new Stack<>();//辅助栈，用来存放最小值
	
	public void push(int node) {
		if(minStack.isEmpty()){
			minStack.push(node);//第一个不用比较就是最小值，直接push
		}else{
			int min = min();
			if(min <= node){
				minStack.push(min);
			}else{
				minStack.push(node);
			}
		}
		dataStack.push(node);
	}

	public void pop() {
		if(minStack.isEmpty()){
			throw new IllegalArgumentException("栈为空");
		}
		minStack.pop();
		dataStack.pop();
	}

	public int top() {
		return minStack.peek();
	}

	public int min() {
		if(minStack.isEmpty())
			throw new IllegalArgumentException("栈为空");
		return top();
	}  
}
```

#### 21、栈的压入、弹出序列
题目描述：

> 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

解题思路：
建立一个辅助栈s,先将序列1,2,3,4,5依次压入栈s，每次压栈时都判断栈s的当前栈顶元素跟序列4,5,3,2,1的第一个元素是否相等。当压入4之后，发现栈顶元素跟序列4,5,3,2,1的第一个元素相等。弹出栈s的栈顶元素4，然后将序列4,5,3,2,1中第一个元素去掉，序列4,5,3,2,1 变成序列5,3,2,1。继续执行上述过程
```javascript
public boolean isPopOrder(int [] pushA,int [] popA) {
		if(pushA.length <= 0)
			return false;
		Stack<Integer> stack = new Stack<>();
		int k=0;
		for(int i=0; i<pushA.length; i++){
			stack.push(pushA[i]);
			if(pushA[i] == popA[k]){
				stack.pop();
				k++;
			}
		}
		while(!stack.isEmpty()){
			if(stack.pop() != popA[k++]){
				return false;
			}
		}
		return true;
    }
```

#### 22、从上往下打印二叉树
题目描述：

> 从上往下打印出二叉树的每个节点，同层节点从左至右打印。

解题思路：
```javascript
public ArrayList<Integer> printFromTopToBottom(TreeNode root) {
		//思路：利用树的层序遍历，用ArrayList做辅助队列
		ArrayList<Integer> list = new ArrayList<>();
		ArrayList<TreeNode> queue = new ArrayList<>();
		if (root == null) {
			return list;
		}
		queue.add(root);
		while (queue.size() != 0) {
			TreeNode temp = queue.remove(0);//弹出第一个节点
			if (temp.left != null) {
				queue.add(temp.left);
			}
			if (temp.right != null) {
				queue.add(temp.right);
			}
			list.add(temp.val);
		}
		return list;
	}
```

#### 23、二叉搜索树的后序遍历序列
题目描述：

> 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

解题思路：
二叉搜索树是根结点大于左子树，小于右子树
在后序遍历序列中，最后一个数字是树的根节点的值。数组中前面的数字可以分为两部分：第一部分是左子树节点的值,它们都比根节点的值小；第二部分是右子树节点的值，它们都比根节点的值大。
	所以先取数组中最后一个数，作为根节点。然后从数组开始计数比根节点小的数，并将这些记作左子树，然后判断后序的树是否比根节点大，
	如果有点不满足，则跳出，并判断为不成立。全满足的话，依次对左子树和右子树递归判断。
```javascript
public boolean verifySquenceOfBST(int[] sequence) {
		if(sequence == null || sequence.length == 0)
			return false;
		
		return process(sequence, 0, sequence.length-1);
	}
	
	public boolean process(int[] sequence,int start,int end) {
		if(start > end)
			return true;
		int root = sequence[end];//最后一个是根结点
		int i = start;
		while(sequence[i] < root){//左子树都小于根结点
			i++;
		}
		int j=i;
		while(j<end){
			if(sequence[j] < root){//若有一个右子树小于（题意没有相等的值）根结点，返回false
				return false;
			}
			j++;
		}
		//递归，看左右子树是否也符合
		boolean left = process(sequence,start,i-1);
		boolean right = process(sequence,i, end-1);
		return left && right;
	}
```

#### 24、二叉树中和为某一值的路径
题目描述：

> 输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

解题思路：
```javascript
     ArrayList<ArrayList<Integer>> pathList = new  ArrayList<ArrayList<Integer>>();
	 ArrayList<Integer> path = new ArrayList<>();
	public ArrayList<ArrayList<Integer>> findPath(TreeNode root, int target) {
		if(root == null)
			return pathList;
		path.add(root.val);
		if(root.left==null && root.right==null && root.val==target){//根到叶节点的值的和刚好=target，则将该路径加到list中
			pathList.add(new ArrayList<Integer>(path));//不重新new的话从始至终pathList中所有引用都指向了同一个path
		}
		if(root.left!=null){
			findPath(root.left, target-root.val);
		}
		if(root.right!=null){
			findPath(root.right, target-root.val);
		}
		path.remove(path.size()-1);//回退到父节点
		return pathList;
	}
```

#### 25、复杂链表的复制
题目描述：

> 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

解题思路：
```javascript
class RandomListNode {
    int label;
    RandomListNode next = null;
    RandomListNode random = null;

    RandomListNode(int label) {
        this.label = label;
    }
}
```
```javascript
  public RandomListNode clone(RandomListNode pHead) {
		HashMap<RandomListNode,RandomListNode> map = new HashMap<RandomListNode,RandomListNode>();
		RandomListNode cur = pHead;
		while(cur != null){
			map.put(cur, new RandomListNode(cur.label));//原节点为key，复制节点为value
			cur = cur.next;
		}
		cur = pHead;
		while(cur != null){
			map.get(cur).next = map.get(cur.next);//复制节点的next指向原节点的next的复制节点
			map.get(cur).random = map.get(cur.random);//复制节点的random指向原节点的random的复制节点
			cur = cur.next;
		}
		return map.get(pHead);
    }
```
#### 26、二叉搜索树与双向链表
题目描述：

> 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190115223949697.png)

解题思路：中序遍历
```javascript
private TreeNode pre = null;
private TreeNode head = null;

public TreeNode Convert(TreeNode root) {
    inOrder(root);
    return head;
}

private void inOrder(TreeNode node) {
    if (node == null)
        return;
    inOrder(node.left);
    node.left = pre;
    if (pre != null)
        pre.right = node;
    pre = node;
    if (head == null)
        head = node;
    inOrder(node.right);
}
```

#### 27、字符串的排列
题目描述：

> 输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

解题思路：
```javascript
    //回溯法
	public ArrayList<String> permutation(String str) {
		ArrayList<String> res = new ArrayList<String>();
		if(str != null && str.length() > 0){
			process(str.toCharArray(), 0, res);
			Collections.sort(res);
		}
		return res;
    }
	
	public void process(char[] str,int i,ArrayList<String> list){
		if(i == str.length-1){
			String val = String.valueOf(str);
			if(!list.contains(val)){
				list.add(val);
			}
		}else{
			for(int j=i; j<str.length; j++){
				swap(str,i,j);
				process(str, i+1, list);
				swap(str,i,j);
			}
		}
	}
	
	public void swap(char[] str,int i,int j){
		char temp = str[i];
		str[i] = str[j];
		str[j] = temp;
	}
```

#### 28、数组中出现次数超过一半数字
题目描述：

> 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

解题思路：
```javascript
public int moreThanHalfNum_Solution(int [] array) {
		int len = array.length;
		if(array == null || len == 0)
			return 0;
		int count = 1,cur = array[0];
		for(int i=1; i<array.length; i++){
			if(array[i] == cur){//遇到相同元素，count++;
				count++;
			}else{
				count--;//遇到不相同元素,count--
			}
			if(count == 0){//当遇到count为0的情况，又以新的i值作为当前值，继续循环
				cur = array[i];
				count = 1;
			}
		}
		count = 0;
		//计算cur值出现的次数
		for(int j=0; j<len; j++){
			if(array[j] == cur)
				count++;
		}
		if(count*2 > len)//出现次数大于数组长度一半即为所求
			return cur;
		
		return 0;
    }
```

#### 29、最小的k个数
题目描述：

> 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

解题思路：
```javascript
    public ArrayList<Integer> getLeastNumbers_Solution(int[] input, int k) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		if(input.length==0 || k>input.length || k<=0)
			return list; 
		//创建能存放k个数的大根堆
		PriorityQueue<Integer> maxHeap = new PriorityQueue<>(k, new Comparator<Integer>(){
			@Override
			public int compare(Integer o1, Integer o2) {
				return o2.compareTo(o1);
			}
			
		} );
		for(int i=0; i<input.length; i++){
			if(maxHeap.size() < k){
                		maxHeap.offer(input[i]);
            		}else if(maxHeap.peek() > input[i]){
                		int temp = maxHeap.poll();
                		maxHeap.offer(input[i]);
            		}
		}
		for (Integer integer : maxHeap) {
			list.add(integer);
		}
		
		return list;
	}
```

#### 30、连续子数组的最大和
题目描述：

> HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

解题思路：
```javascript
  //动态规划
	public int findGreatestSumOfSubArray(int[] array) {
		if(array == null || array.length == 0)
			return 0;
		int maxSum = array[0], maxCur = array[0];
		for(int i=1; i<array.length; i++){
			maxCur = Math.max(array[i], maxCur+array[i]);
			maxSum = Math.max(maxCur, maxSum);
		}
		return maxSum;
	}
	//法二
	public int findGreatestSumOfSubArray2(int[] array) {
		int max = Integer.MIN_VALUE,sum=0;
		for(int i=0; i<array.length; i++){
			if(sum < 0){
				sum = array[i];//记录当前最大值
			}else{
				sum += array[i];//当array[i]为正数时，加上之前的最大值并更新最大值。
			}
			
			if(sum > max)
				max = sum;
		}
		return max;
	}
```
#### 31、从1到n整数中1出现的次数
题目描述：

> 求出1到13的整数中1出现的次数,并算出100到1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

解题思路：
这道题的思路有点绕，需要总结规律，我也不是很懂，有兴趣的同学可以去牛客或者博客找找别人写的详细解答
```javascript
public int NumberOf1Between1AndN_Solution(int n) {
    int cnt = 0;
    for (int m = 1; m <= n; m *= 10) {
        int a = n / m, b = n % m;
        cnt += (a + 8) / 10 * m + (a % 10 == 1 ? b + 1 : 0);
    }
    return cnt;
}
```
#### 32、把数组排成最小的数
题目描述：

> 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

解题思路：
```javascript
    public String PrintMinNumber(int [] numbers) {
        String res = "";
		if(numbers == null || numbers.length == 0 )
			return res;
		//将int数组转化为String数组
		String[] str = new String[numbers.length];
		for (int i=0; i<numbers.length; i++) {
			str[i] = String.valueOf(numbers[i]);
		}
		//自定义排序规则，两个字符串拼接之后谁小谁放前面，例如例如数组{3,32,321}，3和32排序过程是：332与323比较，323较小，故而32排在3之前，
		//同理，321和3排序，3213<321,所以321排在3之前，321和32排序，得到321排在32之前，
		//故而排完之后数组变为{321,32,3}；后面直接拼接数组即可得到答案
		Arrays.sort(str,new Comparator<String>() {
			@Override
			public int compare(String o1, String o2) {
				return (o1+o2).compareTo(o2+o1);
			}
		});
		//排序后的字符数组直接拼接起来就是最小数字
		for(int i=0; i<str.length; i++){
			res += str[i];
		}
		return res;
    }
```
#### 33、丑数
题目描述：

> 把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

解题思路：
丑数的定义是1或者因子只有2 3 5，可推出丑数=丑数*丑数，假定丑数有序序列为：a1,a2,a3.......an
	所以可以将以上序列（a1除外）可以分成3类，必定满足：
	包含2的有序丑数序列：2*a1, 2*a2, 2*a3 .....
	包含3的有序丑数序列：3*a1, 3*a2, 3*a3 .....
	包含5的有序丑数序列：5*a1, 5*a2, 5*a3 .....
	以上3个序列的个数总数和为n个，而且已知a1 = 1了，将以上三个序列合并成一个有序序列即可 
	程序中num2,num3,num5实际就是合并过程中三个序列中带排序的字段索引
```javascript
public int getUglyNumber_Solution(int index) {
		if(index < 7){//1到6都是丑数
			return index;
		}
		int[] res = new int[index];
		res[0] = 1;
		int num2=0,num3=0,num5=0;
		for(int i=1; i<index; i++){
			res[i] = Math.min(res[num2]*2, Math.min(res[num3]*3, res[num5]*5));
			if(res[i] == res[num2]*2)
				num2++;
			if(res[i] == res[num3]*3)
				num3++;
			if(res[i] == res[num5]*5)
				num5++;
		}	
        return res[index-1];
    }
```
#### 34、第一个只出现一次的字符
题目描述：

> 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.

解题思路：
```javascript
    public int firstNotRepeatingChar(String str) {
		HashMap<Character,Integer> map = new HashMap<>();
		if(str == null || str.length() == 0)
			return -1;
		int len = str.length();
		for(int i=0; i<len; i++){
			if(map.containsKey(str.charAt(i))){
				int value = map.get(str.charAt(i));
				map.put(str.charAt(i), value+1);
			}else{
				map.put(str.charAt(i), 1);
			}
		}
		for(int i=0; i<len; i++){
			if(map.get(str.charAt(i)) == 1){//按str顺序遍历得到第一个只出现一次的字符
				return i;
			}
		}
		return -1;
	}
```
#### 35、数组中的逆序对
题目描述：

> 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

解题思路：
归并排序
```javascript
public int inversePairs(int [] array) {
		if(array == null || array.length < 2)
			return 0;
		int res = mergeSort(array, 0, array.length-1);
		return res%1000000007;
    }
	public int mergeSort(int[] array,int L,int R){
		if(L == R){
			return 0;
		}
		int mid = L + (R-L)/2;
		return mergeSort(array,L,mid)+mergeSort(array,mid+1,R)+merge(array,L,mid,R);
	}
	public int merge(int[] array,int L,int mid,int R){
		int[] help = new int[R-L+1];
		int p1=L,p2=mid+1,i=0,res=0;
		while(p1 <= mid && p2 <= R){
			res += array[p1] > array[p2] ? (mid-p1+1) : 0;//计算逆序对
			help[i++] = array[p1] < array[p2] ? array[p1++] : array[p2++];
		}
		while(p1 <= mid){
			help[i++] = array[p1++];
		}
		while(p2 <= R){
			help[i++] = array[p2++];
		}
		for(int j=0; j<help.length; j++){
			array[L+j] = help[j];
		}
		
		return res;
	}		
```
#### 36、两个链表的第一个公共结点
题目描述：

> 输入两个链表，找出它们的第一个公共结点。

解题思路：
```javascript
public ListNode findFirstCommonNode(ListNode pHead1, ListNode pHead2) {
		if(pHead1 == null || pHead2 == null){
			return null;
		}
		ListNode cur1=pHead1,cur2=pHead2;
		int n = 0;
		while(cur1.next != null){
			n++;//计算链表1的长度
			cur1 = cur1.next;
		}
		while(cur2.next != null){
			n--;//遍历完成之后n就是链表1、2的长度差
			cur2 = cur2.next;
		}
		if(cur1 != cur2){
			return null;
		}
		cur1 = n > 0 ? pHead1 : pHead2;//cur1指向长度较长的链表
		cur2 = cur1==pHead1 ? pHead2 : pHead1;
		n = Math.abs(n);
		while(n != 0){//长链表先走n步
			n--;
			cur1 = cur1.next;
		}
		while(cur1 != cur2){//两链表同时每次走一步，相遇时停止
			cur1 = cur1.next;
			cur2 = cur2.next;
		}
		return cur1;//返回相遇的结点
    }
```
#### 37、数字在排序数组中出现的次数
题目描述：

> 统计一个数字在排序数组中出现的次数。

解题思路：
二分查找
```javascript
public int GetNumberOfK(int[] nums, int K) {
    int first = binarySearch(nums, K);
    int last = binarySearch(nums, K + 1);
    return (first == nums.length || nums[first] != K) ? 0 : last - first;
}

private int binarySearch(int[] nums, int K) {
    int low = 0, high = nums.length;
    while (low  < high ) {
        int mid = low  + (high - l) / 2;
        if (nums[mid] >= K)
            high  = mid;
        else
            low  = mid + 1;
    }
    return low ;
}
```
#### 38、二叉树的深度
题目描述：

> 输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

解题思路：
```javascript
  public int TreeDepth(TreeNode root) {
        if(root == null){
			return 0;
		}
		
		int left = TreeDepth(root.left);
		int right = TreeDepth(root.right);
		
		return 1+Math.max(left, right);
    }
```
#### 39、平衡二叉树
题目描述：

> 输入一棵二叉树，判断该二叉树是否是平衡二叉树。

解题思路：
```javascript
//从下往上遍历，如果子树是平衡二叉树，则返回子树的高度；如果发现子树不是平衡二叉树，则直接停止遍历，这样至多只对每个结点访问一次
   public boolean IsBalanced_Solution(TreeNode root) {
        return getDepth(root)!=-1;
    }
    private int getDepth(TreeNode root){
		if(root == null) 
			return 0;
		int left = getDepth(root.left);
		if(left == -1)
			return -1;
		int right = getDepth(root.right);
		if(right == -1)
			return -1;
		return Math.abs(left-right)>1 ? -1 : 1+Math.max(left, right);
	}
```
#### 40、数组中只出现一次的数字
题目描述：

> 一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。请写程序找出这两个只出现一次的数字。
> //num1,num2分别为长度为1的数组。传出参数
//将num1[0],num2[0]设置为返回结果

解题思路：
位运算中异或的性质：两个相同数字异或=0，一个数和0异或还是它本身。
		当只有一个数出现一次时，我们把数组中所有的数，依次异或运算，最后剩下的就是落单的数，因为成对儿出现的都抵消了。
		依照这个思路，我们来看两个数（我们假设是AB）出现一次的数组。我们首先还是先异或，剩下的数字肯定是A、B异或的结果，
		这个结果的二进制中的1，表现的是A和B的不同的位。我们就取第一个1所在的位数，假设是第3位，接着把原数组分成两组，分组标准是第3位是否为1。
		如此，相同的数肯定在一个组，因为相同数字所有位都相同，而不同的数，肯定不在一组。然后把这两个组按照最开始的思路，依次异或，
		剩余的两个结果就是这两个只出现一次的数字。
```javascript
public void findNumsAppearOnce(int[] array, int[] num1, int[] num2) {
		int length = array.length;
		if (length == 2) {
			num1[0] = array[0];
			num2[0] = array[1];
			return;
		}
		int bitResult = 0;
		for (int i = 0; i < length; ++i) {
			bitResult ^= array[i];
		}
		int index = findFirst1(bitResult);
		for (int i = 0; i < length; ++i) {
			if (isBit1(array[i], index)) {
				num1[0] ^= array[i];
			} else {
				num2[0] ^= array[i];
			}
		}
	}

	private int findFirst1(int bitResult) {
		int index = 0;
		while (((bitResult & 1) == 0) && index < 32) {
			bitResult >>= 1;
			index++;
		}
		return index;
	}

	private boolean isBit1(int target, int index) {
		return ((target >> index) & 1) == 1;
	}
```
#### 41、和为S的连续正数序列
题目描述：

> 小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

解题思路：
```javascript
 public ArrayList<ArrayList<Integer> > findContinuousSequence(int sum) {
		 ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		 if(sum <= 0 )
			 return res;
		 int low = 1,high = 2;
		 while(low < high){
			 int cur = (high+low)*(high-low+1)/2;//序列和：（首项加末项）x 项数 / 2;
			 if(cur < sum){
				 high++;
			 }
			 if(cur == sum){
				 ArrayList<Integer> list = new ArrayList<Integer>();
				 for(int i=low; i<=high; i++){
					 list.add(i);
				 }
				 res.add(list);
				 low++;
			 }
			 if(cur > sum){
				 low++;
			 }
		 }
		 return res;
	 }
```
#### 42、和为S的两个数字
题目描述：

> 输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

解题思路：
```javascript
//递增排序--》左右逼近
	public ArrayList<Integer> findNumbersWithSum(int [] array,int sum) {
		ArrayList<Integer> res = new ArrayList<Integer>(2);
		if(array == null || array.length < 2)
			return res;
		int low=0,high=array.length-1;
		while(low < high){
			if(array[low]+array[high] == sum){
				res.add(array[low]);
				res.add(array[high]);
				break;
			}else if(array[low]+array[high] < sum){
				low++;
			}else{
				high--;
			}
		}
		
		return res;
    }
```
#### 43、左旋转字符串
题目描述：

> 汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

解题思路：
```javascript
/*经典的三次翻转：
	1.先翻转字符串前n个字符；
	2.再翻转后面的字符；
	3.翻转整个字符串；
	比如：输入字符串s="12345abc"，n=5;首先翻转前5个字符变成54321abc，然后翻转后面的字符变成54321cba，最后翻转整个字符串变成abc12345*/
	 public String leftRotateString(String str,int n) {
		if(str==null || str.length()<2)
			return str;
		char[] ch = str.toCharArray();
		n = n%ch.length;
		reverse(ch, 0, n-1);
		reverse(ch, n, ch.length-1);
		reverse(ch, 0, ch.length-1);
		return new String(ch);
	        
	 }
	 public void reverse(char[] ch,int start,int end){
		 while(start < end){
			 char temp = ch[start];
			 ch[start++] = ch[end];
			 ch[end--] = temp;
			 //start++;
			 //end--;
		 }
	 }
```
#### 44、翻转单词顺序列
题目描述：

> 牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

解题思路：
```javascript
   public String reverseSentence(String str) {
		if(str == null || str.length() < 2)
			return str;
		char[] data = str.toCharArray();
		reverse(data, 0, data.length-1);//翻转整个句子，I am a student.-->.tneduts a ma I
		int start=0,end=0;
		while(start < data.length){//翻转句子中的每个单词,.tneduts a ma I-->student. a am I
			if(data[start] == ' '){//若起始字符为空格，则begin和end都自加
				start++;
				end++;
			}else if(end==data.length || data[end]==' '){
				reverse(data, start, end-1);//找到一个单词，翻转
				end++;
				start = end;
			}else{//没有遍历到空格或者遍历结束，则单独对end自增
				end++;
			}
		}
		return String.valueOf(data);

	}
	
	private void reverse(char[] data,int start,int end){
		while(start < end){
			char temp = data[start];
			data[start++] = data[end];
			data[end--] = temp;
		}
	}
```
#### 45、扑克牌顺子
题目描述：

> LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。

解题思路：
```javascript
  public boolean isContinuous(int [] numbers) {
		if(numbers==null || numbers.length!=5)
			return false;
		Arrays.sort(numbers);
		int zero = 0, num = 0;
		for(int i=0; i<numbers.length-1; i++){
			if(numbers[i] == 0){//计算大小王数量
				zero++;
				continue;
			}
			if(numbers[i] == numbers[i+1]){//重复则不可能是顺子
				return false;
			}
			//加和后，得到的是元素差的总和，因为等差数列差是1，所以就可以算做差零的个数，最后判断if (Zero >= num) 
			num += numbers[i+1]-numbers[i]-1;
		}
		if(zero >= num){
			return true;
		}
		return false;

	 }
```
#### 46、圆圈中最后剩下的数
题目描述：

> 让小朋友们围成一个大圈。然后，随机指定一个数 m，让编号为 0 的小朋友开始报数。每次喊到 m-1 的那个小朋友要出列唱首歌，然后可以在礼品箱中任意的挑选礼物，并且不再回到圈中，从他的下一个小朋友开始，继续 0...m-1 报数 .... 这样下去 .... 直到剩下最后一个小朋友，可以不用表演。

解题思路：
约瑟夫环，圆圈长度为 n 的解可以看成长度为 n-1 的解再加上报数的长度 m。因为是圆圈，所以最后需要对 n 取余。
```javascript
public int LastRemaining_Solution(int n, int m) {
    if (n == 0)     /* 特殊输入的处理 */
        return -1;
    if (n == 1)     /* 递归返回条件 */
        return 0;
    return (LastRemaining_Solution(n - 1, m) + m) % n;
}
```
#### 47、求1+2+3+...+n
题目描述：

> 求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

解题思路：
```javascript
public int Sum_Solution(int n) {
       int sum = n;
		boolean flag = (sum>0) && (sum+=Sum_Solution(n-1))>0;
		 
		return sum; 
    }
```
#### 48、不用加减乘除做加法
题目描述：

> 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

解题思路：

```javascript
     /*我们可以用三步走的方式计算二进制值相加： 5-101，7-111
	 * 第一步：相加各位的值，不算进位，得到010，二进制每位相加就相当于各位做异或操作，101^111。
	 * 第二步：计算进位值，得到1010，相当于各位做与操作得到101，再向左移一位得到1010，(101&111)<<1。
	 * 第三步重复上述两步， 各位相加 010^1010=1000，进位值为100=(010&1010)<<1。 继续重复上述两步：1000^100 =
	 * 1100，进位值为0，跳出循环，1100为最终结果。
	 */
	/*
	 * 两个数异或：相当于每一位相加，而不考虑进位；
	 *  两个数相与，并左移一位：相当于求得进位； 
	 *  将上述两步的结果相加
	 * */
public int Add(int num1,int num2) {
       while(num2 != 0){
			int sum = num1 ^ num2;
			int carray = (num1 & num2) << 1;
			num1 = sum;
			num2 = carray;
		}
		return num1; 
    }
```
#### 49、把字符串转换为整数
题目描述：

> 将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。
> 示例1
输入
+2147483647
    1a33
输出
2147483647
    0

解题思路：
```javascript
public int strToInt(String str) {
		boolean isPosotive = true;
		char[] arr = str.toCharArray();
		int sum = 0;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == '+') {
				continue;
			} else if (arr[i] == '-') {
				isPosotive = false;
				continue;
			} else if (arr[i] < '0' || arr[i] > '9') {
				return 0;
			}
			sum = sum * 10 + (int) (arr[i] - '0');
		}
		if (sum == 0 || sum > Integer.MAX_VALUE || sum < Integer.MIN_VALUE) {
			return 0;
		}
		return isPosotive == true ? sum : -sum;
	}
```
#### 50、数组中重复的数字
题目描述：

> 在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

解题思路：


```javascript
这种数组元素在 [0, n-1] 范围内的问题，可以将值为 i 的元素调整到第 i 个位置上。
以 (2, 3, 1, 0, 2, 5) 为例：
position-0 : (2,3,1,0,2,5) // 2 <-> 1
             (1,3,2,0,2,5) // 1 <-> 3
             (3,1,2,0,2,5) // 3 <-> 0
             (0,1,2,3,2,5) // already in position
position-1 : (0,1,2,3,2,5) // already in position
position-2 : (0,1,2,3,2,5) // already in position
position-3 : (0,1,2,3,2,5) // already in position
position-4 : (0,1,2,3,2,5) // nums[i] == nums[nums[i]], exit

public boolean duplicate(int numbers[],int length,int [] duplication) {
    	if(numbers == null || length==0)
    		return false;
    	for(int i=0; i<length; i++){
    		while(numbers[i]!=i){
    			if(numbers[i] == numbers[numbers[i]]){//true-->重复
					duplication[0] = numbers[i];
					return true;
    			}
    			int temp = numbers[i];
    			numbers[i] = numbers[temp];
    			numbers[temp] = temp;
    		}
    	}
		return false;
    
    }
```
#### 51、构建乘积数组
题目描述：

> 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

解题思路：
```javascript
public int[] multiply(int[] A) {
    int n = A.length;
    int[] B = new int[n];
    for (int i = 0, product = 1; i < n; product *= A[i], i++)       /* 从左往右累乘 */
        B[i] = product;
    for (int i = n - 1, product = 1; i >= 0; product *= A[i], i--)  /* 从右往左累乘 */
        B[i] *= product;
    return B;
}
```
#### 52、正则表达式匹配
题目描述：

> 请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

解题思路：
```javascript

```
#### 53、
题目描述：

解题思路：
```javascript
/*
	 * 当模式中的第二个字符不是“*”时： 
	 * 1、如果字符串第一个字符和模式中的第一个字符相匹配，那么字符串和模式都后移一个字符，然后匹配剩余的。 
	 * 2、如果字符串第一个字符和模式中的第一个字符相不匹配，直接返回false。
	 * 而当模式中的第二个字符是“*”时：
	 * 如果字符串第一个字符跟模式第一个字符不匹配，则模式后移2个字符，继续匹配。如果字符串第一个字符跟模式第一个字符匹配，可以有3种匹配方式：
	 * 1、模式后移2字符，相当于x*被忽略；
	 * 2、字符串后移1字符，模式后移2字符；
	 * 3、字符串后移1字符，模式不变，即继续匹配字符下一位，因为*可以匹配多位；
	 * 
	 * 这里需要注意的是：Java里，要时刻检验数组是否越界。
	 */
	public boolean match(char[] str, char[] pattern){
		if(str == null || pattern == null){
			return false;
		}
		int sIndex = 0;//str的index
		int pIndex =0;
		
		return process(str,pattern,sIndex,pIndex);
    }
	
	public boolean process(char[] str,char[] pattern,int sIndex,int pIndex){
		if(sIndex == str.length && pIndex == pattern.length){//str到尾，pattern到尾，匹配成功
			return true;
		}
		if(sIndex!=str.length && pIndex==pattern.length){//pattern先到尾，匹配失败
			return false;
		}
		//模式第2个是*，且字符串第1个跟模式第1个匹配,分3种匹配模式；如不匹配，模式后移2位
		if(pIndex+1<pattern.length && pattern[pIndex+1]=='*'){
			if((sIndex != str.length && pattern[pIndex] == str[sIndex]) || (pattern[pIndex] == '.' && sIndex != str.length)){
				return process(str,pattern,sIndex,pIndex+2)//模式后移2，视为x*匹配0个字符
					   || process(str,pattern,sIndex+1,pIndex+2)//视为模式匹配1个字符
					   || process(str,pattern,sIndex+1,pIndex);//*匹配1个，再匹配str中的下一个
			}else{
				return process(str,pattern,sIndex,pIndex+2);
			}
		}
		//模式第2个不是*，且字符串第1个跟模式第1个匹配，则都后移1位，否则直接返回false
		if((sIndex!=str.length && pattern[pIndex]==str[sIndex]) || (pattern[pIndex] == '.' && sIndex != str.length)){
			return process(str,pattern,sIndex+1,pIndex+1);
		}
		
		return false;
	}
```
#### 54、表示数值的字符串
题目描述：

> 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

解题思路：
```javascript
public boolean isNumeric(char[] str) {
		if(str==null || str.length==0)
			return false;
		if(str.length==1){
			if((str[0]!='e' || str[0]=='E') || (str[0]<'0' && str[0]>'9'))
				return false;
		}
		 // 标记符号、小数点、e是否出现过
		boolean sign=false, decimal=false, hasE=false;
		for(int i=0; i<str.length; i++){
			if(str[i]=='e' || str[i]=='E'){
				if(i==str.length-1)// e后面一定要接数字
					return false; 
				if(hasE)  // 不能同时存在两个e
					return false;
				hasE = true;
			}else if(str[i]=='+' || str[i]=='-'){
				if(sign && str[i-1]!='e' && str[i-1]!='E')// 第二次出现+-符号，则必须紧接在e之后
					return false;
				if(!sign && i>0 && str[i-1]!='e' && str[i-1]!='E')// 第一次出现+-符号，且不是在字符串开头，则也必须紧接在e之后
					return false;
				sign = true;
			}else if(str[i] == '.'){
				if(hasE || decimal) // e后面不能接小数点，小数点不能出现两次
					return false;
				decimal = true;
			}else if(str[i]<'0' && str[i]>'9'){//不能出现无效字符
				return false;
			}
		}
		
		return true;
    }
```
#### 55、字符流中第一个不重复的字符
题目描述：

> 请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
> 如果当前字符流没有存在出现一次的字符，返回#字符。

解题思路：
```javascript
//法一，一个字符占8位，因此不会超过256个，可以申请一个256大小的数组来实现一个简易的哈希表。时间复杂度为O(n)，空间复杂度O(n).
	StringBuilder s = new StringBuilder();
	char[] hash = new char[256];
	 //Insert one char from stringstream
    public void Insert(char ch) {
        s.append(ch);
        hash[ch]++;
    }
    //return the first appearence once char in current stringstream
    public char FirstAppearingOnce(){
    	//int len = s.length();
    	char[] str = s.toString().toCharArray();
    	for(int i=0; i<str.length; i++){
    		if(hash[str[i]]==1){
    			return str[i];
    		}
    			
    	}
		return '#';   
    }
    
    //法二，利用LinkedHashMap的有序性
    Map<Character,Integer> map = new LinkedHashMap<>();
    public void Insert2(char ch) {
       if(map.containsKey(ch)){
    	   map.put(ch, map.get(ch)+1);
       }else{
    	   map.put(ch, 1);
       }
    }
    //return the first appearence once char in current stringstream
    public char FirstAppearingOnce2(){
    	for (Map.Entry<Character, Integer> c : map.entrySet()) {
			if(c.getValue()==1){
				return c.getKey();
			}
		}
    	return '#';
    }
```
#### 56、链表中环的入口结点
题目描述：

> 给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null

解题思路：
快慢指针思想
```javascript
if(pHead==null || pHead.next==null || pHead.next.next==null)
			return null;
		ListNode slow = pHead.next;
		ListNode fast = pHead.next.next;
		while(slow != fast){
			if(fast.next==null || fast.next.next==null){
				return null;
			}
			slow = slow.next;
			fast = fast.next.next;
		}
		fast = pHead;
		while(slow != fast){
			slow = slow.next;
			fast = fast.next;
		}
		return slow;

	}
```
#### 57、删除链表中重复的结点
题目描述：

解题思路：
```javascript
//递归
	public ListNode deleteDuplication(ListNode pHead){
		if(pHead == null || pHead.next==null)
			return pHead;
		//ListNode cur = pHead;
		if (pHead.val == pHead.next.val) { // 当前结点是重复结点
			ListNode pNode = pHead.next;
			while (pNode != null && pNode.val == pHead.val) {
				// 跳过值与当前结点相同的全部结点,找到第一个与当前结点不同的结点
				pNode = pNode.next;
			}
			return deleteDuplication(pNode); // 从第一个与当前结点不同的结点开始递归
		} else { // 当前结点不是重复结点
			pHead.next = deleteDuplication(pHead.next); // 保留当前结点，从下一个结点开始递归
			return pHead;
		}

    }
```
#### 58、二叉树的下一个结点
题目描述：

> 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

解题思路：
```javascript
class TreeLinkNode {
    int val;
    TreeLinkNode left = null;
    TreeLinkNode right = null;
    TreeLinkNode next = null;//指向父节点的指针

    TreeLinkNode(int val) {
        this.val = val;
    }
}
```
```javascript
/* 最暴力的解法是直接通过指向父节点的指针找到根结点，然后中序遍历直接返回该结点的下一个结点，但这种方法复杂度较高，不提倡
 * 以如下树为例（中序遍历顺序为4251637）中序遍历顺序的下一个结点也称后继结点，有如下规律
 * 一、如果一个点有右孩子，那么它的后继结点就是它右子树上的最左的结点（比如2有右孩子，它的后继结点就是5,1有右孩子，后继结点就是右子树367上最左的6）
 * 二、如果一个点x没有右孩子，
 * (1)通过next找到x的父节点，若x是其父的左孩子，那么后继结点就是这个父节点（比如4无右，是2的左孩子，后继就是2）
 * (2)通过next找到x的父节点，若x是其父的右孩子，则继续往上找其父的父节点，直到某个点是其父节点的左孩子，那么后继结点就是这个父节点
 * （5无右，是2的右孩子，继续往上，2的父为1,2是1的左孩子，后继就是1）（7往上一直找到1,1可以看成一个null的左，故后继就是null）
 * 
 *     1
 *   2   3
 *  4 5 6 7  
 * */
public class GetNext {
	public TreeLinkNode getNext(TreeLinkNode pNode) {
		if(pNode == null)
			return pNode;
		if(pNode.right != null){//如果有右孩子，返回右子树中最左的结点
			return getLeftMost(pNode.right);
		}else{//如果没有右孩子
			TreeLinkNode parent = pNode.next;
			while(parent != null && parent.left != pNode){//如果当前节点不是其父的左，继续循环，否则直接返回父
				pNode = parent;
				parent = pNode.next;
			}
			return parent;
		}
		
	}
	
	public TreeLinkNode getLeftMost(TreeLinkNode node){
		if(node == null){
			return node;
		}
		while(node.left != null){
			node = node.left;
		}
		return node;
	}
```
#### 59、对称的二叉树
题目描述：

> 请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

解题思路：
```javascript
//法一，递归
	boolean isSymmetrical(TreeNode pRoot){
		if(pRoot == null){
			return true;
		}
		
		return process(pRoot.left,pRoot.right);
    }
	public boolean process(TreeNode left,TreeNode right){
		if(left == null && right == null){
			return true;
		}
		if(left == null || right == null){//当左右有一者为空，自然不可能对称
			return false;
		}
		if(left.val!=right.val){//左右值不等，直接返回false
			return false;
		}
		return process(left.left,right.right)//左孩子的左孩子与右孩子的右孩子是否对称
			   && process(left.right,right.left);//左孩子的右孩子与右孩子的左孩子是否对称
	}
	//法二，非递归算法，利用DFS
	boolean isSymmetricalDFS(TreeNode pRoot){
		if(pRoot == null)
			return true;
		Stack<TreeNode> stack = new Stack<TreeNode>();
		stack.push(pRoot.left);
		stack.push(pRoot.right);
		while(!stack.isEmpty()){
			TreeNode right = stack.pop();//取出左右节点，注意是右节点先出
			TreeNode left = stack.pop();
			if(left==null && right==null){
				continue;
			}
			if(left==null || right==null){
				return false;
			}
			if(left.val!=right.val){
				return false;
			}
			stack.push(left.left);
			stack.push(right.right);
			stack.push(left.right);
			stack.push(right.left);
		}
		return true;
	}
```
#### 60、按之字形打印二叉树
题目描述：

> 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

解题思路：
```javascript
public class Solution {
    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
        Stack<TreeNode> stack1 = new Stack<TreeNode>();
        Stack<TreeNode> stack2 = new Stack<TreeNode>();
        int layer = 1;
        stack1.push(pRoot);
        while(!stack1.isEmpty() || !stack2.isEmpty()){
            ArrayList<Integer> list = new ArrayList<Integer>();
            if(layer%2==1){
                while(!stack1.isEmpty()){
                    TreeNode node = stack1.pop();
                    if(node != null){
                        list.add(node.val);
                        stack2.push(node.left);
                        stack2.push(node.right);
                    }
                }
            }else{
                while(!stack2.isEmpty()){
                    TreeNode node = stack2.pop();
                    if(node != null){
                        list.add(node.val);
                        stack1.push(node.right);
                        stack1.push(node.left);
                    }
                }
            }
            if(!list.isEmpty()){
                res.add(list);
                layer++;
            }
        }
        return res;
    }
}
```
#### 61、把二叉树打印成多行
题目描述：

> 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

解题思路：
```javascript
 //递归
	ArrayList<ArrayList<Integer>> Print2(TreeNode pRoot) {
		ArrayList<ArrayList<Integer>> list = new ArrayList<>();
		depth(pRoot, 1, list);
		return list;
	}

	private void depth(TreeNode root, int depth, ArrayList<ArrayList<Integer>> list) {
		if (root == null)
			return;
		if (depth > list.size())
			list.add(new ArrayList<Integer>());
		list.get(depth - 1).add(root.val);

		depth(root.left, depth + 1, list);
		depth(root.right, depth + 1, list);
	}
```
#### 62、序列化二叉树
题目描述：

> 请实现两个函数，分别用来序列化和反序列化二叉树

解题思路：
算法思想：根据前序遍历规则完成序列化与反序列化。所谓序列化指的是遍历二叉树为字符串；所谓反序列化指的是依据字符串重新构造成二叉树。
 * 依据前序遍历序列来序列化二叉树，因为前序遍历序列是从根结点开始的。当在遍历二叉树时碰到Null指针时，这些Null指针被序列化为一个特殊的字符“#”。
 * 另外，结点之间的数值用逗号隔开。
```javascript
   //序列化
	String Serialize(TreeNode root) {
		StringBuilder sb = new StringBuilder();
		if(root == null){
			sb.append("#,");
			return sb.toString();
		}
		sb.append(root.val+",");
		sb.append(Serialize(root.left));
		sb.append(Serialize(root.right));
		return sb.toString();
	}
	//反序列化
	TreeNode Deserialize(String str) {
		String[] values = str.split(",");
		Queue<String> queue = new LinkedList<>();
		for(int i=0; i<values.length; i++){
			queue.offer(values[i]);
		}
		return reconPreOrder(queue);
	}
	public TreeNode reconPreOrder(Queue<String> queue){
		String value = queue.poll();
		if(value.equals("#")){
			return null;
		}
		TreeNode node = new TreeNode(Integer.valueOf(value));
		node.left = reconPreOrder(queue);
		node.right = reconPreOrder(queue);
		return node;
	}
```
#### 63、二叉搜索树的第k个结点
题目描述：

> 给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。

解题思路：
```javascript
    //递归      
    // 思路：二叉搜索树按照中序遍历的顺序打印出来正好就是排序好的顺序。
	//所以，按照中序遍历顺序找到第k个结点就是结果。
	/*if (node != null)
				return node;
	如果没有if(node != null)这句话  那么那个root就是返回给上一级的父结点的，而不是递归结束的条件了,
	有了这句话过后，一旦返回了root，那么node就不会为空了，就一直一层层的递归出去到结束了。
	举第一个例子{8,6,5,7,},1 答案应该是5，如果不加的时候，开始，root=8，node=kth（6,1），
	继续root=6，node=kth（5,1）root =5 返回null，这时向下执行index=k=1了，返回 5给root=6递归的时候的node，
	这时回到root=8了，往后面调右孩子的时候为空而把5给覆盖了。现在就为空了，有了这句话后虽然为空，但不把null返回，而是继续返回5，*/
	int index = 0; // 计数器
	TreeNode kthNode(TreeNode pRoot, int k) {
		if (pRoot != null) { // 中序遍历寻找第k个
			TreeNode node = kthNode(pRoot.left, k);
			if (node != null)
				return node;
			index++;
			if (index == k)
				return pRoot;
			node = kthNode(pRoot.right, k);
			if (node != null)
				return node;
		}
		return null;
	}
	//非递归
	TreeNode kthNode2(TreeNode pRoot, int k){
		if(pRoot==null || k<=0){
			return null;
		}
		Stack<TreeNode> stack = new Stack<TreeNode>();
		TreeNode cur = pRoot;
		int count = 0;
		while(cur!=null || !stack.isEmpty()){
			if(cur != null){
				stack.push(cur);
				cur = cur.left;
			}else{
				cur = stack.pop();
				count++;
				if(count == k){
					return cur;
				}
				cur = cur.right;
			}
		}
		return null;
		
	}
```
#### 64、数据流中的中位数
题目描述：

> 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

解题思路：
```javascript
    /*
	 * 思路： 为了保证插入新数据和取中位数的时间效率都高效，这里使用大顶堆+小顶堆的容器，并且满足：
	 * 1、两个堆中的数据数目差不能超过1，这样可以使中位数只会出现在两个堆的交接处； 2、大顶堆的所有数据都小于小顶堆，这样就满足了排序要求。
	 */
	PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();//小根堆
	PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(11,new Comparator<Integer>() {//11是初始容量
		@Override
		public int compare(Integer o1, Integer o2) {
			return o2.compareTo(o1);//反转排序---大根堆
		}
	});
	int count;

	public void Insert(Integer num) {
	    count++;
	    if(count%2 == 0){//若为偶数
	    	if(!maxHeap.isEmpty() && num<maxHeap.peek()){//小的数进大根堆
	    		maxHeap.offer(num);
	    		num = maxHeap.poll();
	    	}
	    	minHeap.offer(num);
	    }else{
	    	if(!minHeap.isEmpty() && num>minHeap.peek()){//大的数进小根堆
	    		minHeap.offer(num);
	    		num = minHeap.poll();
	    	}
	    	maxHeap.offer(num);
	    }
    }

    public Double GetMedian() {
    	if(count == 0){
    		return null;
    	}
    	//总数为奇数时，大顶堆堆顶就是中位数
    	if(count%2 == 1){
    		return new Double(maxHeap.peek());
    	}else{
    		return (maxHeap.peek()+minHeap.peek())/2.0;
    	}
        
    }
```
#### 65、滑动窗口的最大值
题目描述：

> 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

解题思路：
```javascript
//利用双端队列
	public ArrayList<Integer> maxInWindows(int [] num, int size) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		if(num==null || num.length==0 || size<=0 ){
			return list;
		}
		LinkedList<Integer> deque = new LinkedList<Integer>();//双端队列，用来记录每个窗口的最大值下标
		for(int i=0; i<num.length; i++){
			while(!deque.isEmpty() && num[deque.peekLast()]<num[i]){
				deque.pollLast();
			}
			deque.addLast(i);
			if(deque.peekFirst() == i-size){//判断队首元素是否过期
				deque.pollFirst();
			}
			if(i >= size-1){ //向list列表中加入元素
				list.add(num[deque.peekFirst()]);
			}
		}
		return list;
	}
```

#### 66、矩阵中的路径
题目描述：

> 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

解题思路：
```javascript
/*
	 * 回溯 基本思想： 0.根据给定数组，初始化一个标志位数组，初始化为false，表示未走过，true表示已经走过，不能走第二次
	 * 1.根据行数和列数，遍历数组，先找到一个与str字符串的第一个元素相匹配的矩阵元素，进入judge
	 * 2.根据i和j先确定一维数组的位置，因为给定的matrix是一个一维数组
	 * 3.确定递归终止条件：越界，当前找到的矩阵值不等于数组对应位置的值，已经走过的，这三类情况，都直接false，说明这条路不通
	 * 4.若k，就是待判定的字符串str的索引已经判断到了最后一位，此时说明是匹配成功的
	 * 5.下面就是本题的精髓，递归不断地寻找周围四个格子是否符合条件，只要有一个格子符合条件，就继续再找这个符合条件的格子的四周是否存在符合条件的格子，
	 * 直到k到达末尾或者不满足递归条件就停止。 6.走到这一步，说明本次是不成功的，我们要还原一下标志位数组index处的标志位，进入下一轮的判断。
	 */
	public boolean hasPath(char[] matrix, int rows, int cols, char[] str){
		boolean[] flag = new boolean[matrix.length];
		for(int i=0; i<rows; i++){
			for(int j=0; j<cols; j++){
				//遍历二维数组，找到第一个等于str的第一个字符的值，再递归判断
				if(process(matrix, rows, cols, i, j, 0, flag, str)){
					return true;
				}
			}
		}
		return false;
    }
	//矩阵、行数、列数、行坐标、列坐标、待判断的字符位、标记flag数组、原字符串
	public boolean process(char[] matrix,int rows,int cols,int i,int j,int k,boolean[] flag, char[] str){
		int index = i*cols+j;//先根据i和j计算匹配的第一个元素转为一维数组的位置
		//递归终止条件
		if (i < 0 || j < 0 || i >= rows || j >= cols || matrix[index] != str[k] || flag[index] == true){
			return false;
		}
		//若k已经到达str末尾了，说明之前的都已经匹配成功了，直接返回true即可
		if(k == str.length-1){
			return true;
		}
		//走过的位置置为true，表示已经走过了
		flag[index] = true;
		//回溯，递归寻找，每次找到了就给k加一，找不到，还原
		if(process(matrix, rows, cols, i-1, j, k+1, flag, str)||//i-1-->向上
		   process(matrix, rows, cols, i+1, j, k+1, flag, str)||//i+1-->向下
		   process(matrix, rows, cols, i, j-1, k+1, flag, str)||//j-1-->向左
		   process(matrix, rows, cols, i, j+1, k+1, flag, str)){//j+1-->向右
			return true;
		}
		
		//走到这，说明这一条路不通，还原，再试其他的路径
        flag[index] = false;
        return false;
	}
```
#### 67、机器人的运动范围
题目描述：

> 地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

解题思路：
```javascript
 public int movingCount(int threshold, int rows, int cols){
		int[][] flag = new int[rows][cols];
		return process(threshold, rows, cols, 0, 0, flag);
	 }
	 public int process(int threshold,int rows,int cols,int i,int j,int[][] flag){
		 if(threshold<1 || i<0 || j<0 || i>=rows || j>=cols || flag[i][j]==1 || (sum(i)+sum(j)>threshold)){
			 return 0;
		 }
		 flag[i][j] = 1;//标记已走过
		 int res = process(threshold, rows, cols, i-1, j, flag) +
				   process(threshold, rows, cols, i+1, j, flag) +
				   process(threshold, rows, cols, i, j-1, flag) +
				   process(threshold, rows, cols, i, j+1, flag) + 1;
		 return res;
	 }
	 //计算各数位之和
	 public int sum(int i){
		 if(i==0){
			 return i;
		 }
		 int sum = 0;
		 while(i != 0){
			 sum += i%10;
			 i /=10;
		 }
		 return sum;
	 }
```


# ​Top 100

## 01 [3Sum](https://leetcode.com/problems/3sum/)

> 给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
> 注意：答案中不可以包含重复的三元组。例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，
> 满足要求的三元组集合为：
> [
>   [-1, 0, 1],
>   [-1, -1, 2]
> ]

```java
public class Solution{
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        for(int i=0; i<nums.length-2; i++){
            if(i>0 && nums[i] == nums[i-1])
                continue;
            for(int j=i+1,k=nums.length-1; j<k; ){
                if(nums[i]+nums[j]+nums[k] == 0){
                    res.add(Arrays.asList(nums[i],nums[j],nums[k]));
                    j++;
                    k--;
                    while(j<k && nums[j] == nums[j-1])
                        j++;
                    while(j<k && nums[k] == nums[k+1])
                        k--;
                }else if(nums[i]+nums[j]+nums[k] > 0){
                    k--;
                }else{
                    j++;
                }
            }
        }
        return res;
    }
}
```

## 02 [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)

> 给定两个非空链表来表示两个非负整数。位数按照逆序方式存储，它们的每个节点只存储单个数字。将两数相加返回一个新的链表。
> 你可以假设除了数字 0 之外，这两个数字都不会以零开头。
> 示例：
> 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
> 输出：7 -> 0 -> 8
> 原因：342 + 465 = 807

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode p1=l1,p2=l2,cur=dummy;
        int carry = 0;
        while(p1!=null || p2!=null){
            int x = p1!=null ? p1.val : 0;
            int y = p2!=null ? p2.val : 0;
            int sum = x+y+carry;
            carry = sum/10;
            cur.next = new ListNode(sum%10);
            cur = cur.next;
            if(p1!=null)
                p1 = p1.next;
            if(p2 != null)
                p2 = p2.next;
        }
        if(carry > 0){
            cur.next = new ListNode(carry);
        }
        return dummy.next;       
    }
}
```

## 03 [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

> 假设您有一个数组，其中第*i* 个元素是第*i*天给定股票的价格。如果您只被允许完成最多一笔交易（即买入并卖出一股股票），请设计一个算法来查找最大利润。请注意，在购买之前不能出售股票。
>
> **例1：**
>
> ```
> 输入： [7,1,5,3,6,4]
>  输出： 5
>  说明：在第2天买入（价格= 1）并在第5天卖出（价格= 6），利润= 6-1 = 5。
>              不是7-1 = 6，因为售价需要大于购买价格。
> ```
>
> **例2：**
>
> ```
> 输入： [7,6,4,3,1]
>  输出： 0
>  说明：在这种情况下，没有进行任何交易，即最大利润= 0。
> ```

```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices==null || prices.length<2)
            return 0;
        int max=0,curMin=prices[0];
        for(int i=0; i<prices.length; i++){
            if(prices[i] > curMin){
                max = Math.max(max,prices[i]-curMin);
            }else{
                curMin = prices[i];
            }
        }
        return max;
    }
}
```

## 04 [Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

> 假设有一个数组，其中第i个元素是给定股票在第i天的价格。
> 设计一种寻找最大利润的算法。你可以完成任意多笔交易(例如，多次买卖一股股票)，但有以下限制:
> 你不能同时进行多项交易(即，你必须先卖出股票，然后再买)。
> 你卖掉股票后，第二天就不能买股票了。(即冷却1天)
> 例子:
> 输入:(1, 2, 3 ,0 ,2]
> 输出:3
> 说明:交易=[买入，卖出，冷却，买入，卖出]

```java
class Solution {
    //此法过于复杂，难以理解
    public int maxProfit(int[] prices) {
        if(prices== null || prices.length<2)
            return 0;
        int len = prices.length;
        int[] s0 = new int[len];
        int[] s1 = new int[len];
        int[] s2 = new int[len];
        s0[0] = 0;
        s1[0] = -prices[0];
        s2[0] = Integer.MIN_VALUE;
        for(int i=1; i<len; i++){
            s0[i] = Math.max(s0[i-1], s2[i-1]);
            s1[i] = Math.max(s1[i-1], s0[i-1]-prices[i]);
            s2[i] = Math.max(s2[i-1], s1[i-1]+prices[i]);
        }
        return Math.max(s0[len-1], s2[len-1]);
    }
}
```

## 05 [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

> 给定一个二叉树，返回其节点值的中序遍历。
> 例子:
> 输入:[1,null,2,3]
> 1
> \
> 2
> /
> 3.
> 输出(1、3、2):
> 后续:递归解决方案很简单，您可以迭代地完成它吗?

```java
//递归
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        if(root == null)
            return new ArrayList<>();
        List<Integer> list = new ArrayList<Integer>();
       
        list.addAll(inorderTraversal(root.left));
        list.add(root.val);
        list.addAll(inorderTraversal(root.right));
        return list;
    }
}
//迭代
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        List<Integer> list = new ArrayList<Integer>();
        while(!stack.isEmpty() || root!=null){
            while(root != null){
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            list.add(root.val);
            root = root.right;
        }
        return list;
    }
}
```

## 06 [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

> 给定一个二叉树，返回其节点值的层次遍历。(即从左到右，一层一层地)。
> 例如:
> 给定二叉树[3,9,20,null,null,15,7]，
>
>   3
>    / \
>   9  20
>     /  \
>    15   7
>
> 返回其层次顺序遍历为:
>
> ```
> [
>   [3],
>   [9,20],
>   [15,7]
> ]
> ```

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        if(root == null)
            return res;
        queue.offer(root);
        while(!queue.isEmpty()){
            int levelNum = queue.size();//每层的个数
            List<Integer> list = new ArrayList<Integer>();
            for(int i=0; i<levelNum; i++){
                TreeNode temp = queue.poll();
                if(temp.left != null)
                    queue.offer(temp.left);
                if(temp.right != null)
                    queue.offer(temp.right);
                list.add(temp.val);
            }
            res.add(list);
        }
        return res;      
    }
}
```

## 07 [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

> 给定一个非空二叉树，求最大路径和。
> 对于这个问题，路径被定义为沿着父-子连接从某个起始节点到树中的任意节点的任意节点序列。路径必须包含至少一个节点，并且不需要经过根节点。
> 示例:
>
> ```
> Input: [1,2,3]
>        1
>       / \
>      2   3
> Output: 6
> ```
>
> ```
> 
> Input: [-10,9,20,null,null,15,7]
>    -10
>    / \
>   9  20
>     /  \
>    15   7
> Output: 42
> ```

```java
class Solution {
    public int maxPathSum(TreeNode root) {
        int[] res = new int[1];
        res[0] = Integer.MIN_VALUE;
        findMax(root,res);
        return res[0];
    }
    public static int findMax(TreeNode root,int[] res){
        if(root == null)
            return 0;
        int left = findMax(root.left,res);
        int right = findMax(root.right,res);
        int curSum = Math.max(left,0)+Math.max(right,0)+root.val;
        res[0] = Math.max(res[0],curSum);
        return Math.max(left>0?left:0,right>0?right:0)+root.val;
    }
}
```

## 08 [Burst Balloons](https://leetcode.com/problems/burst-balloons/)

> 给定n个气球，索引从0到n-1。每个气球上都绘有一个由数组号表示的数字。要求你把所有的气球都吹破。如果你的 i 气球爆了，你会得到(左边)*(i)*(右边)的硬币。这里左、右是i的邻指标，经过爆破后，左、右变成邻指标。
> 明智地爆破气球，找到你能收集到的最多的硬币。
> 注意:您可以想象nums[-1] = nums[n] = 1。它们不是真实的，所以你不能把它们炸开。0≤n≤500,0≤nums[i]≤100
> 例子:
> 输入:(3、1、5、8)
> 输出:167
> 解释:num = (3、1、5、8) - - > (3、5、8) - - > [3 8] - > [8] - > []
> 硬币= 3x1x5 + 3x5x8 + 1x3x8 + 1x8x1 = 167

```java
class Solution {
    public int maxCoins(int[] nums) {
        int len = nums.length;
        int[][] dp = new int[len][len];
        return helper(nums, 0, len-1, dp);
    }
    public static int helper(int[] nums,int start,int end,int[][] dp){
        if(start > end)
            return 0;
        if(dp[start][end] != 0)
            return dp[start][end];
        int max = 0;
        for(int i=start; i<=end; i++){
            int curMax = helper(nums,start,i-1,dp) +
                		 value(nums,start-1)*value(nums,i)*value(nums,end+1) +
                		 helper(nums,i+1,end,dp);
            max = Math.max(max,curMax);
        }
        dp[start][end] = max;
        return max;
    }
    public static int value(int[] nums,int i){
        if(i<0 || i>=nums.length)
            return 1;
        return nums[i];
    }
}
```

## 09 [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

> 你正在爬楼梯,到达山顶需要n步。每次你可以爬1或2级台阶。你可以用几种不同的方式爬到山顶? 注意:给定n是一个正整数。
> 示例1:
> 输入:2
> 输出:2
> 说明:爬到山顶有两种方法。
>
> 1.1步+ 1步
>
> 2.2步
>
> 示例2:
> 输入:3
> 输出:3
> 说明:爬到山顶有三种方法。
>
> 1. 1步+ 1步+ 1步
> 2. 1步+ 2步
> 3. 2步+ 1步

```java
class Solution {
    public int climbStairs(int n) {
        if(n <= 0)
            return 0;
        if(n == 1 || n == 2)
            return n;
        int res=0,f1=1,f2=1;
        for(int i=1; i<n; i++){
            res = f1+f2;
            f1 = f2;
            f2 = res;
        }
        return res;
    }
}
```

## 10 [Coin Change](https://leetcode.com/problems/coin-change/)

> 你会得到不同面值的硬币和总额。写一个函数来计算你需要的最少的硬币数量。如果那笔钱不能由硬币的任何组合来补足，返回-1。
> 示例1:
> 输入:硬币=[1,2,5]，金额= 11
> 输出:3
> 说明:11 = 5 + 5 + 1
> 示例2:
> 输入:硬币=[2]，金额= 3
> 输出:-1
> 注意:
> 你可以假设每种硬币的数量都是无限的。

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        if(coins==null || coins.length==0 || amount<=0)
            return 0;
        int[] dp = new int[amount+1];
        Arrays.fill(dp,amount+1);
        dp[0] = 0;
        for(int i=1; i<dp.length; i++){
            for(int j=0; j<coins.length; j++){
                if(coins[j] <= i){
                    dp[i] = Math.min(dp[i],dp[i-coins[j]]+1);
                }
            }
        }
        return dp[amount]==(amount+1) ? -1 : dp[amount];
    }
}
```

## 11 [Combination Sum](https://leetcode.com/problems/combination-sum/)

> 给定一组候选编号(候选人)(没有重复)和一个目标编号(目标)，找出候选编号与目标编号之和的所有唯一组合。
> 同一重复号码可从候选编号中无限次选择。
> 注意:所有的数字(包括目标)都是正整数。解集不能包含重复的组合。
> 示例1:
> 输入:candidate = [2,3,6,7]， target = 7，
> 解集为:
> [
> [7],
> (2、2、3)
> ]
> 示例2:
> 输入:candidate = [2,3,5]， target = 8，
> 解集为:
> (
> (2、2、2、2),
> (2、3、3),
> (3、5)
> ]

```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        int len = candidates.length;
        List<List<Integer>> res = new ArrayList<ArrayList<Integer>>();
        List<Integer> list = new ArrayList(Integer)();
        if(len == 0)
            return res;
        Arrays.sort(candidates);
        dfs(res,list,candidates,0,0,target);
        return res;
    }
    public void dfs(List<List<Integer>> res,List<Integer> list,int[] candidates,
                    int start,int sum,int target){
        if(sum == target){
            res.add(new ArrayList<Integer>(list));
            return;
        }
        for(int i=start; i<candidates.length; i++){
            sum += candidates[i];
            if(sum > target)
                return;
            list.add(candidates[i])
            dfs(res,list,candidates,i,sum.target);
            list.remove(list.size()-1);
            sum -= candidates[i];
        }
    }
}
```

## 12  [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

> 给定树的前序序和中序，构造二叉树。
> 注意:您可以假设树中不存在重复结点。
> 例如,给定
>
> ```
> preorder = [3,9,20,15,7]
> inorder = [9,3,15,20,7]
> ```
>
> Return the following binary tree:
>
> ```
>     3
>    / \
>   9  20
>     /  \
>    15   7
> ```

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder.length==0 || inorder.length==0)
            return null;
        return build(preorder,inorder,0,preorder.length-1,0,inorder.length-1);
    }
    public TreeNode build(int[] pre,int[] in,int pStart,int pEnd,
                          int iStart,int iEnd){
        TreeNode tree = new TreeNode(pre[pStart]);//根结点
        tree.left = null;
        tree.right = null;
        if(pStart==pEnd && iStart==iEnd)
            return tree;
        int root = 0;
        for(root=iStart; root<iEnd; root++){
            if(pre[pStart] == in[root]){//找到inorder中根结点的位置
                break;
            }
        }
        int left = root-iStart;
        int right = iEnd-root;
        if(left > 0){
            tree.left = build(pre,in,pStart+1,pStart+left,iStart,root-1);
        }
        if(right > 0){
            tree.right = build(pre,in,pStart+left+1,pEnd,root+1,iEnd);
        }
        return tree;
    }
}
```

## 13 [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

```java
class Solution {
    public int maxArea(int[] height) {
        int low=0,high=height.length-1;
        int max=0;
        while(low < high){
            int lowMax=height[low],highMax=height[high];
            int water = (high-low)*(Math.min(lowMax,highMax));
            max = Math.max(max,water);
            if(height[low] <= height[high]){
                while(low < high && height[low] <= lowMax)
                    low++;
            }else{
                while(low < high && height[high] <= highMax)
                    high--;
            }
        }
        return max;
    }
}
```

## 14 [Convert BST to Greater Tree](https://leetcode.com/problems/convert-bst-to-greater-tree/)

> 将二叉搜索树转为较大树，通过题目汇总的例子可以明白，是把每个结点值加上所有比它大的结点值总和当作新的结点值。
>
> ```
> Input: The root of a Binary Search Tree like this:
>               5
>             /   \
>            2     13
> 
> Output: The root of a Greater Tree like this:
>              18
>             /   \
>           20     13
> ```

```java
class Solution {
    public TreeNode convertBST(TreeNode root) {
        int sum = 0;
        process(root,sum);
        return root;
    }
    public void process(TreeNode cur,int sum){
        if(cur != null){
            process(cur.right,sum);
            cur.val += sum;
            sum = cur.val;
            process(cur.left,sum);            
        }
    }
}
```

## 15 [Counting Bits](https://leetcode.com/problems/counting-bits/)

> 给定一个非负整数数字num，对于0≤i≤num范围内的每个数字i，计算其二进制表示形式中1的个数，并将其作为数组返回。
> 示例1:
> 输入:2
> 输出(0,1,1):
> 示例2:
> 输入:5
> 输出:[0,1,1、2、1、2)
> 跟进:
> 用运行时O(n*sizeof(integer))提出一个解决方案是非常容易的。但你能在线性时间O(n)内完成吗?
> 空间复杂度应该是O(n)

```java
class Solution {
    public int[] countBits(int num) {
        int[] res = new int[num+1];
        for(int i=num; i>=0; i--){
        	res[i] = numsofone(num--);
        }
        return res;
    }
    public int numsofone(int num){
        int count = 0;
        while(num != 0){
            num = num & (num-1);
            count++;
        }
        return count;
    }
}
```

## 16  [Course Schedule](https://leetcode.com/problems/course-schedule/)

```java
先略过、、、、、、
```

## 17 [Decode String](https://leetcode.com/problems/decode-string/)

> 给定一个编码字符串，返回它的解码字符串。
> 编码规则是:k[encoded_string]，其中方括号内的encoded_string正好重复k次。注意，k保证是一个正整数。
> 您可以假设输入字符串总是有效的;没有额外的空格，方括号格式良好，等等。
> 此外，您可以假设原始数据不包含任何数字，并且这些数字只针对那些重复的数字，比如k。例如，不会有像3a或2[4]这样的输入。
> 例子:
> s = "3[a]2[bc]"，返回"aaabcbc"。
> s = "3[a2[c]]"，返回"accaccacc"。
> s = "2[abc]3[cd]ef"，返回"abcabccdcdcdef"。

```java
class Solution {
   public String decodeString(String s) {
	String res = "";
	Stack<Integer> countStack = new Stack<>();// 记录'['之前的数字
	Stack<String> strStack = new Stack<>();// 记录'['之前的运算结果
	int index = 0;
	int curNum = 0;
	while(index < s.length()){
		char ch = s.charAt(index);
		if(Character.isDigit(ch)){//如果是数字
			while(Character.isDigit(s.charAt(index))){
				curNum = curNum*10 + (s.charAt(index)-'0');
				index++;
			}
		}else if(ch == '['){
			strStack.push(res);
			countStack.push(curNum);
			res = "";
			curNum = 0;
			index++;
		}else if(ch == ']'){
			StringBuilder temp = new StringBuilder(strStack.pop());
			int len = countStack.pop();
			for(int i=0; i<len; i++){
				temp.append(res);
			}
			res = temp.toString();
			index++;
		}else{//若为字符
			res += s.charAt(index);
			index++;
		}
	}
	return res;
  }
}
```

## 18 [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)

> 给定一棵二叉树，你需要计算树的直径的长度。二叉树的直径是树中任意两个节点之间最长路径的长度。这条路径可以通过根，也可以不通过根。
> 例子:
> 给定一个二叉树
>
> ```
>           1
>          / \
>         2   3
>        / \     
>       4   5    
> 返回3，这是路径的长度[4,2,1,3]或[5,2,1,3]。
> 注:两个节点之间的路径长度由节点之间的边数表示。
> ```

```java
class Solution {
    int max = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        depth(root);
        return max;
    }
    public int depth(TreeNode root){
        if(root == null)
            return 0;
        int left = depth(root.left);
        int right = depth(root.right);
        max = Math.max(max,left+right);
        return Math.max(left,right)+1;
    }
}
```

## 19 [Edit Distance](https://leetcode.com/problems/edit-distance/)

> 给出两个单词word1和word2，找出将word1转换为word2所需的最小操作数。
> 你可以对一个单词进行以下三种操作:
> 插入一个字符
> 删除一个字符
> 替换一个字符
>
> 示例1:
> 输入:word1 = "horse"， word2 = "ros"
> 输出:3
> 解释:
> horse -> rorse(将“h”替换为“r”)
> rorse ->rose(去掉“r”)
> rose -> ros(去掉'e')

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length(), len2 = word2.length();
        int[] dp = new int[len1 + 1][len2 + 1];
        for(int i=0; i<=len1; i++){
            dp[i][0] = i;
        }
        for(int j=0; j<=len2; j++){
            dp[0][j] = j;
        }
        
        for(int i=1; i<=len1; i++){
            char c1 = word1.charAt(i-1);
            for(int j=1; j<=len2; j++){
                char c2 = word2.charAt(j-1);
                if(c1 == c2){
                    dp[i][j] = dp[i-1][j-1];
                }else{
                    int replace = dp[i-1][j-1]+1;
                    int insert = dp[i-1][j]+1;
                    int delete = dp[i][j-1]+1;
                    
                    int min = Math.min(replace,Math.min(insert,delete));
                    dp[i][j] = min;
                }
            }
        }
        return dp[len1][len2];
    }
}
```

## 20 [Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

> 给定一个字符串s和一个非空字符串p，在s中找到所有p的序列的起始索引。
> 字符串只由小写英文字母组成，字符串s和p的长度都不大于20,100。
> 输出的顺序无关紧要。
> 示例1:
> 输入:
> s: "cbaebabacd" p: "abc"
> 输出:
> (0,6)
> 解释:
> 从 index = 0的子字符串是“cba”，它是“abc”的一个序列。
> 从 index = 6的子字符串是“bac”，它是“abc”的一个序列。

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<Integer>();
        if(s==null || s.length()==0 || p.length()==0)
            return res;
        int[] map = new int[256];
        for(char c : p.toCharArray()){
            map[c]++;
        }
        int left=0, right=0, count=p.length();//滑动窗口思想
        while(right < s.length(){
            if(map[s.charAt(right++)]-- > 0)
                count--;
            if(count == 0)
                res.add(left);
            if(right-left == p.length() && map[s.charAt(left++)]++ >= 0)
                count++;
        }
        return res;
    }
}
```

## 21 [Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)

> 给定一个整数数组，其中1≤a[i]≤n (n =数组大小)，有些元素出现两次，有些元素出现一次。
> 找到[1,n]中不出现在该数组中的所有元素。
> 你能在没有额外空间的情况下在O(n)运行时完成吗?您可以假设返回的列表不算作额外的空间。
> 例子:
> 输入:
> [4、3、2、7、8、2、3、1]
> 输出:
> (5、6)

```java
class Solution {
    // 0   1   2   3  4  5   6   7      下标
    // 4   3   2   7  8  2   3   1      原数组
    // -4 -3  -2  -7  8  2  -3  -1     变化后的数组
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<Integer>();
        for(int i=0; i<nums.length; i++){
            int cur = Math.abs(nums[i])-1;//cur当下标，故要减一
            if(nums[cur] > 0){
                nums[cur] = -nums[cur];//将出现的元素置为负数
            }
        }
        for(int i=0; i<nums.length; i++){
            if(nums[i] > 0){
                res.add(i+1);
            }
        }
        return res;
    }
}
```

## 22 [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

> 给定一个按升序排序的整数数组，找出给定目标值的起始和结束位置。
> 算法的运行时复杂度必须是O(log n)的顺序。
> 如果数组中没有找到目标，返回[- 1,-1]。
> 示例1:
> 输入:nums = [5,7,7,8,8,10]， target = 8
> 输出:(3、4)
> 示例2:
> 输入:nums = [5,7,7,8,8,10]， target = 6
> 输出:[-1,-1]

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] res = new int[2];
        res[0] = findFirst(nums,target);
        res[1] = findLast(nums,target);
        return res;
    }
    public int findFirst(int[] nums,int target){
        int low = 0, high = nums.length-1, index = -1;
        while(low <= high){
            int mid = low + (high-low)/2;
            if(nums[mid] >= target){
                high = mid-1;
            }else{
                low = mid+1;
            }
            if(nums[mid] == target){
                index = mid;
            }
        }
        return index;
    }
     public int findLast(int[] nums,int target){
        int low = 0, high = nums.length-1, index = -1;
        while(low <= high){
            int mid = low + (high-low)/2;
            if(nums[mid] <= target){
                low = mid+1;      
            }else{
               high = mid-1; 
            }
            if(nums[mid] == target){
                index = mid;
            }
        }
        return index;
    }
}
```

## 23 [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

> 给定一个包含n + 1个整数的数组，其中每个整数都在1和n之间，证明至少有一个重复的数字必须存在。假设只有一个重复的数字，找出重复的那个。
> 示例1:
> 输入:(1、3、4、2、2]
> 输出:2
> 示例2:
> 输入:[3,1,3,4,2)
> 输出:3
> 注意:
> 不能修改数组(假设数组是只读的)。
> 你必须只使用常数O(1)额外的空间。
> 您的运行时复杂度应该小于O(n2)。
> 数组中只有一个重复的数字，但它可以重复多次。

```java
class Solution {
    //思路： 二分查找
    //可以用二分法先选取n/2，按照抽屉原理，整个数组中如果小于等于n/2的数的数量大于n/2，说明1到n/2这个区间是肯定有重     //复数字的。比如6个抽屉，如果有7个袜子要放到抽屉里，那肯定有一个抽屉至少两个袜子。
    public int findDuplicate(int[] nums) {
        int low=0,high=nums.length-1;
        while(low < high){
            int mid = low + (high - low)/2;
            int minCount=0,equalCount=0;
            for(int i=0; i<nums.length; i++){
                if(nums[i] < mid)
                    minCount++;
                if(nums[i] == mid)
                    equalCount++;
            }
            if(equalCount > 1){//说明重复的数即是mid
                return mid;
            }else if(minCount+equalCount > mid){//说明重复的数在前一半区间
                high = mid -1;
            }else{
                low = mid+1;
            }
        }
        return low;
    }
}
```

## 24 [Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)

> Given a binary tree, flatten it to a linked list in-place.
>
> For example, given the following tree:
>
> ```
>     1
>    / \
>   2   5
>  / \   \
> 3   4   6
> ```
>
> The flattened tree should look like:
>
> ```
> 1
>  \
>   2
>    \
>     3
>      \
>       4
>        \
>         5
>          \
>           6
> ```

```java
class Solution {
    public void flatten(TreeNode root) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode cur = root;
        while(cur != null || !stack.isEmpty()){
            if(cur.right != null)
                stack.push(cur.right);
            if(cur.left != null){
                cur.right = cur.left;
                cur.left = null;
            }else if(!stack.isEmpty()){
                cur.right = stack.pop();
            }
            cur = cur.right;
        }
    }
}
```

## 25 [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)

> 给定n对括号，编写一个函数来生成所有格式良好的括号组合。
> 例如，给定n = 3，解集为:
> (
> ”((()))”,
> ”(()())”,
> ”(())()”,
> “()(())”,
> ”()()()”
> ]

```java
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<String>();
        if(n <= 0)
            return res;
        String str = "";
        dfs(res,str,n,n);
        return res;
    }
    public void dfs(List<String> res, String str, int left, int right){
        if(left > right)//防止出现")("非法括号
            return;
        if(left==0 && right==0){
            res.add(str);
            return;
        }
        if(left > 0)
            dfs(res,str+"(",left-1,right);
        if(right > 0)
            dfs(res,str+")",left,right-1);
    }
}
```

## 26 [Group Anagrams](https://leetcode.com/problems/group-anagrams/)

> Given an array of strings, group anagrams together.
>
> **Example:**
>
> ```
> Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
> Output:
> [
>   ["ate","eat","tea"],
>   ["nat","tan"],
>   ["bat"]
> ]
> 注意：
> 所有输入都是小写的。
> 输出的顺序无关紧要。
> ```

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> res = new ArrayList<ArrayList<String>>();
        HashMap<String,List<String>> map = new HashMap<String,List<String>>();
        if(strs==null || strs.length==0)
            return res;
        String tmp = "";
        for(int i=0; i<strs.length; i++){
            char[] ch = strs[i].toCharArray();
            Arrays.sort(ch);
            tmp = String.valueOf(ch);
            if(map.containsKey(tmp)){
                map.get(tmp).add(strs[i]);
            }else{
                List<String> list = new ArrayList<String>();
                list.add(strs[i]);
                map.put(tmp,list);
            }
        }
       for(List<String> list : map.values()){
       		res.add(list);                        
       }
       return res;
    }  
}
```

## 27 [Hamming Distance](https://leetcode.com/problems/hamming-distance/)

> 两个整数之间的 Hamming 距离是对应位不同位置的个数。
> 给定两个整数x和y，计算 Hamming 距离。
>
> **Note:**
> 0 ≤ `x`, `y` < 231.
>
> **Example:**
>
> ```
> Input: x = 1, y = 4
> 
> Output: 2
> 
> Explanation:
> 1   (0 0 0 1)
> 4   (0 1 0 0)
>           ↑   ↑
> 
> 上面的箭头指向对应位不同的位置。
> ```

```java
class Solution {
    public int hammingDistance(int x, int y) {
        int n = x ^  y;//异或->n中1的数量即为对应不同位置的个数
        int count = 0;
        while(n != 0){
            n = n&(n-1);//每次去掉最低位的1
            count++;
        }
        return count;
    }
}
```

## 28  [House Robber](https://leetcode.com/problems/house-robber/)

> 你是一个计划沿街抢劫房屋的职业强盗。每个房子都有一定数量的钱被藏起来，阻止你抢劫他们的唯一限制是相邻的房子都有安全系统连接，如果两个相邻的房子在同一天晚上被闯入，它会自动联系警察。
> 给出一个非负整数列表，表示每户人家的钱数，确定你今晚不报警就能抢劫的最大钱数。
> 示例1:
> 输入:(1、2、3、1)
> 输出:4
> 说明:抢劫第1家(钱= 1)，然后抢劫第3家(钱= 3)。
> 你可以抢劫的总数= 1 + 3 = 4。
> 示例2:
> 输入:[2、7、9、3、1]
> 输出:12
> 说明:抢房子1(钱= 2)，抢房子3(钱= 9)和抢房子5(钱= 1)。
> 你可以抢劫的总数= 2 + 9 + 1 = 12。

```java
class Solution {
    public int rob(int[] nums) {
        int len = nums.length;
        if(nums == null || len==0)
            return 0;
        if(len == 1)
            return nums[0];
        int[] dp = new int[len];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0],nums[1]);
        for(int i=2; i<len; i++){
            dp[i] = Math.max(dp[i-1],dp[i-2]+nums[i]);
        }
        return res[len-1];
    }
}
```

## 29 [House Robber III](https://leetcode.com/problems/house-robber-iii/)

> 小偷又为他的偷窃行为找到了一个新的地方。这个区域只有一个入口，叫做“根”。除了根，每个房子都有且只有一个父房子。经过一番参观，聪明的小偷意识到“这个地方所有的房子都是二叉树”。如果两个直系房屋在同一天晚上被闯入，它将自动与警方联系。
> 确定小偷今晚不报警就能抢劫的最大金额。
>
> **Example 1:**
>
> ```
> Input: [3,2,3,null,3,null,1]
> 
>      3
>     / \
>    2   3
>     \   \ 
>      3   1
> 
> Output: 7 
> 说明:小偷可以抢劫的最大金额= 3 + 3 + 1 = 7。
> ```
>
> **Example 2:**
>
> ```
> Input: [3,4,5,1,3,null,1]
> 
>      3
>     / \
>    4   5
>   / \   \ 
>  1   3   1
> 
> Output: 9
> 说明:小偷可以抢劫的最大金额= 4 + 5 = 9。
> ```

```java
class Solution {
     public int rob(TreeNode root) {
        int[] res = robSub(root);
        return res[1];
    }
    public int[] robSub(TreeNode root){
        if(root == null)
            return new int[2];
        int[] left = robSub(root.left);
        int[] right = robSub(root.right);
        
        int[] res = new int[2];
        res[0] = left[1] + right[1];
        res[1] = Math.max(left[0]+right[0]+root.val, res[0]);
        return res;
    }
}
```

## 30 [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

> 使用insert、search和startsWith方法实现trie(前缀树or字典树)。
> 例子:
> Trie trie = new Trie();
>
> trie.insert("apple");
> trie.search("apple");   // returns true
> trie.search("app");     // returns false
> trie.startsWith("app"); // returns true
> trie.insert("app");   
> trie.search("app");     // returns true
> 注意:
> 您可以假设所有输入都由小写字母a-z组成。
> 所有输入都保证是非空字符串。

```java
class TrieNode{
    TrieNode[] children = new TrieNode[26];
    String item = "";
    public TrieNode(){
        
    }
}
public class Trie {
	private TrieNode root;
    /** Initialize your data structure here. */
    public Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode node = root;
        for(char c : word.toCharArray()){
            if(node.children[c-'a'] == null){
                node.children[c-'a'] = new TrieNode();
            }
            node = node.children[c-'a'];
        }
        node.item = word;       
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode node = root;
         for(char c : word.toCharArray()){
            if(node.children[c-'a'] == null){
                return false;
            }
            node = node.children[c-'a'];
        }
        return node.item.equals(word);
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for(char c : prefix.toCharArray()){
              if(node.children[c-'a'] == null){
                return false;
            }
            node = node.children[c-'a'];
        }
        return true;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
```

## 31 [Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)

> 编写一个程序，找出两个单链表相交的起始点。
>
> 注:
> 如果两个链表没有交集，返回null。
> 在函数返回后，链表必须保留其原始结构。
> 您可以假设在整个链接结构的任何地方都没有环。
> 您的代码最好在O(n)时间内运行，并且只使用O(1)内存

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null)
            return null;
        ListNode c1 = headA, c2 = headB;
        int i = 0,j = 0;
        while(c1 != null){
            i++;
            c1 = c1.next;
        }
        while(c2 != null){
            j++;
            c2 = c2.next;
        }
        int n = Math.abs(i-j);
        c1 = i-j>0 ? headA : headB;
        c2 = c1==headA ? headB : headA;
        while(n != 0){
            n--;
            c1 = c1.next;
        }
        while(c1 != c2){
            c1 = c1.next;
            c2 = c2.next;
        }
        return c1==c2 ? c1 : null;
    }
}
```

## 32  [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)

> 翻转二叉树
>
> **Example:**
>
> Input:
>
> ```
>      4
>    /   \
>   2     7
>  / \   / \
> 1   3 6   9
> ```
>
> Output:
>
> ```
>      4
>    /   \
>   7     2
>  / \   / \
> 9   6 3   1
> ```

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null)
            return root;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }
}
```

## 33 [Jewels and Stones](https://leetcode.com/problems/jewels-and-stones/)

> 字符串J表示宝石的类型，而S表示你拥有的宝石。S中的每个角色都是你拥有的一种石头。你想知道你有多少宝石也是珠宝。
> J中的字母保证是不同的，并且J和S中的所有字符都是字母。字母是区分大小写的，所以“a”被认为是不同于“A”的石头。
> 示例1:
> 输入:J = "aA"， S = "aAAbbbb"
> 输出:3
> 示例2:
> 输入:J = "z"， S = "ZZ"
> 输出:0
> 注意:
> S和J由字母组成，长度不超过50。
> J中的字符是不同的。

```java
class Solution {
    public int numJewelsInStones(String J, String S) {
        int[] map = new int[256];
        int count = 0;
        for(char c : J.toCharArray()){
            map[c]++;
        }
        for(int i=0; i<S.length(); i++){
            if(map[S.charAt(i)] == 1){
                count++;
            }
        }
        return count;
    }
}
```

## 34 [Jump Game](https://leetcode.com/problems/jump-game/)

> 给定一个非负整数数组，初始位置是数组的第一个索引。
> 数组中的每个元素表示该位置的最大跳转长度。
> 确定是否能够达到最后一个索引。
> 示例1:
> 输入:[2、3、1、1、4)
> 输出:true
> 说明:从索引0跳转到1，然后再跳转到最后一个索引，执行3个步骤。
> 示例2:
> 输入:[3、2、1、0,4]
> 输出:false
> 说明:无论如何，您总是会到达索引3。它的最大
> 跳转长度为0，因此不可能到达最后一个索引。

```java
class Solution {
    public boolean canJump(int[] nums) {
        int len = nums.length;
        if(nums==null || len==0)
            return false;
        if(len == 1)
            return true;
        int max = 0;//记录当前点能到达的最大位置
        for(int i=0; i<len-1; i++){
            max = Math.max(max,i+nums[i]);
            if(max < i+1)//当前点的最大跳转长度到不了下一点
                return false;
            if(max >= len-1)//当前点的最大跳转长度能到达最后一点
                return true;    
        }
        return false;
    }
}
```

## 35  [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

> 找出未排序数组中第k大的元素。注意，它是排序后第k大的元素，而不是第k个不同的元素。
> 示例1:
> 输入:[3,2,1,5,6,4]，k = 2
> 输出:5
> 示例2:
> 输入:[3,2,3,1,2,4,5,5,6]，k = 4
> 输出:4
> 注意:
> 你可以假设k总是有效的，1≤k≤数组的长度。

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        if(nums==null || nums.length<k)
            return 0;
        int L = 0, R = nums.length-1, target = nums.length-k;
        while(true){ 
            int index = partition(nums,k,L,R);
            if(index < target){
                L = index+1;
            }else if(index > target){
                R = index-1;
            }else{
                return nums[index];
            }
        }
    }
    public int partition(int[] nums,int k,int L,int R){
       int pivot = nums[L];
        int index = L;
        for(int i=L+1; i<=R; i++){
            if(nums[i] < pivot){
                index++;
                swap(nums,index,i);
            }
        }
        swap(nums,L,index);
        return index;
    }
    public void swap(int[] nums,int i,int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

## 36  [LRU Cache](https://leetcode.com/problems/lru-cache/)

> 为最近最少使用的缓存设计和实现数据结构。它应该支持以下操作:get和put。
> get(key)—如果键存在于缓存中，则获取键的值(总是正的)，否则返回-1。
> put(key, value)——如果键不存在，则设置或插入该值。当缓存达到其容量时，应该在插入新项之前使最近最少使用的项无效。
> 跟进:
> 你能在O(1)时间复杂度下做这两个操作吗?
> 例子:
>
> ```
> LRUCache cache = new LRUCache( 2 /* capacity */ );
> 
> cache.put(1, 1);
> cache.put(2, 2);
> cache.get(1);       // returns 1
> cache.put(3, 3);    // evicts key 2
> cache.get(2);       // returns -1 (not found)
> cache.put(4, 4);    // evicts key 1
> cache.get(1);       // returns -1 (not found)
> cache.get(3);       // returns 3
> cache.get(4);       // returns 4
> ```

```java
class LRUCache {
    static class Node{
        private int key;
        private int value;
        private Node pre;
        private Node next;
        public Node(){}
        public Node(int key,int value){
            this.key = key;
            this.value = value;
        }
    }
    private int capacity;//指定的容量
    private Node head;
    private Node tail;
    private HashMap<Integer,Node> map;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        head = new Node();//初始化头尾节点，这里的头结点是辅助节点,head节点不存储任何有效元素
        tail = head;
        map = new HashMap<Integer,Node>((capacity/0.75)+1);//构造器初始容量这样设置可以保证map不会发生扩容
    }
    //访问一个键值对
    public int get(int key) {
        Node cur = map.get(key);
        if(cur == null){
            return -1;
        }else{
            removeNode(cur);
            addToTail(cur);
            return cur.value;
        }
    }
    //存储一个键值对
    public void put(int key, int value) {
        Node cur = map.get(key);
        if(cur == null){
            cur = new Node(key,value);
            map.put(key,cur);
            addToTail(cur);
            //超出了容量，移除链表头结点后面那个元素(头结点是辅助节点)
            if(map.size()>capacity && head!=tail){
                Node out = head.next;
                removeNode(out);
                map.remove(out.key);
            }
        }else{
            removeNode(cur);
            cur.value = value;
            addToTail(cur);
        }
    }
    //将指定节点追加到链表末尾
    private void addToTail(Node cur){
        tail.next = cur;
        cur.pre = tail;
        tail = cur;
    }
    //将指定节点从链表中删除
    private void removeNode(Node cur){
        if(cur == tail){
            tail = tail.pre;
            tail.next = null;
            cur.pre = null;
        }else{
            cur.pre.next = cur.next;
            cur.next.pre = cur.pre;
            cur.pre = null;
            cur.next = null;
        }
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```

## 37 [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

> 给定n个非负整数表示直方图的条高，其中每个条的宽度为1，求直方图中最大矩形的面积。
>
>  
>
> ![img](https://assets.leetcode.com/uploads/2018/10/12/histogram.png)
> 上面是一个柱状图，给定高度=[2,1,5,6,2,3]，每个柱状图的宽度为1。
>
>  
>
> ![img](https://assets.leetcode.com/uploads/2018/10/12/histogram_area.png)
> 最大的矩形显示在阴影区域，它的面积= 10个单位
>
> **Example:**
>
> ```
> Input: [2,1,5,6,2,3]
> Output: 10
> ```

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        for(int i=0; i<=len; i++){
            int h = (i==len ? 0 : heights[i]);
            if(stack.isEmpty() || h>=heights[stack.peek()]){
                stack.push(i);
            }else{
                int top = stack.pop();
                int temp = heights[top]*(s.isEmpty()?i:i-1-stack.peek());
                maxArea = Math.max(maxArea,temp);
                i--;
            }
        }
        return maxArea;
    }
}
```

## 38 [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

> 给定一个包含2-9个数字的字符串，返回该数字可能表示的所有字母组合。
> 下面给出了数字到字母的映射(就像电话按钮一样)。注意，1不映射到任何字母
>
> 
>
> ![img](http://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Telephone-keypad2.svg/200px-Telephone-keypad2.svg.png)
>
> Example:
>
> Input: "23"
> Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
>
> 注意:
> 尽管上面的答案是按照字典顺序排列的，但是您的答案可以是您想要的任何顺序。

```java
public class Solution {
    List<String> res;    
    public List<String> letterCombinations(String digits) {
        // 建立映射表
        String[] table = {" ", " ", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        StringBuilder tmp = new StringBuilder();
        res = new LinkedList<String>();
        helper(table, 0, tmp, digits);
        return res;
    }
    
    private void helper(String[] table, int idx, StringBuilder tmp, String digits){
        if(idx == digits.length()){
            // 找到一种结果，加入列表中
            if(tmp.length()!=0) res.add(tmp.toString());
        } else {
            // 找出当前位数字对应可能的字母
            String candidates = table[digits.charAt(idx) - '0'];
            // 对每个可能字母进行搜索
            for(int i = 0; i < candidates.length(); i++){
                tmp.append(candidates.charAt(i));
                helper(table, idx+1, tmp, digits);
                tmp.deleteCharAt(tmp.length()-1);
            }
        }
    }
}

```

## 39 [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

> 给定一个链表，判断它是否有一个循环。
>
> 为了在给定的链表中表示一个循环，我们使用一个整数pos，它表示tail连接到的链表中的位置(0索引)。如果pos为-1，则链表中没有循环。
>
> **Example 1:**
>
> ```
> Input: head = [3,2,0,-4], pos = 1
> Output: true
> 说明:链表中有一个循环，tail连接到第二个节点。
> ```
>
> 
>
> ![img](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)

```java
public class Solution {
    //快慢指针思想
    public boolean hasCycle(ListNode head) {
        if(head==null || head.next==null || head.next.next==null)
            return false;
        ListNode slow=head.next,fast=head.next.next;
        while(slow != fast){
            if(fast.next==null || fast.next.next==null){
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }
}
```

## 40 [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)

> 给定一个链表，返回循环开始的节点。如果没有循环，返回null。
>
> 为了在给定的链表中表示一个循环，我们使用一个整数pos，它表示tail连接到的链表中的位置(0索引)。如果pos为-1，则链表中没有循环。
>
> 注意:不要修改链表。
>
> 示例1:
>
> 输入:head = [3,2,0，-4]， pos = 1
> 输出:tail连接到节点索引1
> 说明:链表中有一个循环，tail连接到第二个节点。

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if(head==null || head.next==null || head.next.next==null)
            return null;
        ListNode slow = head.next,fast = head.next.next;
        while(slow != fast){
            if(fast.next == null || fast.next.next == null){
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        fast = head;
        while(slow != fast){
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
}
```

## 41  [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

> 给定一个未排序的整数数组，找出最长的连续元素序列的长度。
>
> 您的算法应该在O(n)复杂度下运行。
>
> 例子:
>
> 输入:[100,4,200,1,3,2]
> 输出:4
> 说明:最长的连续元素序列为[1,2,3,4]。因此它的长度是4。

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        int max = 0;
        for(int i=0; i<nums.length; i++){
            set.add(nums[i]);
        }
        for(int i=0; i<nums.length; i++){
            int len = 0;
            int num = nums[i];
            while(set.contains(num)){
                len++;
                set.remove(num);
                num++;
            }
            num = nums[i]-1;
            while(set.contains(num)){
                len++;
                set.remove(num);
                num--;
            }
            if(len > max)
                max = len;
        }
        return max;
    }
}
```

## 42 [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

> 给定一个未排序的整数数组，求其最长递增子序列的长度。
>
> 例子:
>
> 输入:[10, 9, 2, 5, 3, 7, 101, 18)
> 输出:4
> 说明:最长递增子序列为[2,3,7,101]，因此长度为4。
> 注意:
>
> 可能有多个LIS组合，只需要返回长度即可。
> 您的算法应该在O(n2)复杂度下运行。
> 跟进:你能把它的时间复杂度提高到O(n log n)吗?

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        if(nums==null || nums.length==0)
            return 0;
        int[] dp = new int[nums.length];
        int max = 1;
        for(int i=0; i<nums.length; i++){
            dp[i] = 1; 
            for(int j=0; j<i; j++){
                if(nums[j] < nums[i]){
                    dp[i] = Math.max(dp[i], dp[j]+1);
                }
            }
            if(dp[i] > max){
                max = dp[i];
            }
        }          
        return max;
    }
}
```

## 43 [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

> 	给定一个字符串s，找出s中最长的回文子串。可以假设s的最大长度为1000。
>
> 示例1:
>
> 输入:“babad”
> 输出:“bab”
> 注:“aba”也是一个有效的答案。
> 示例2:
>
> 输入:“cbbd”
> 输出:“bb”

```java
class Solution {
   /* 对于每个子串的中心（可以是一个字符，或者是两个字符的间隙，比如串abc,中心可以是a,b,c,或者是ab的间隙，bc的间隙，例如aba是回文，abba也是回文，这两种情况要分情况考虑）往两边同时进 行扫描，直到不是回文串为止。假设字符串的长度为n,那么中心的个数为2*n-1(字符作为中心有n个，间隙有n-1个）。对于每个中心往两边扫描的复杂 度为O(n),所以时间复杂度为O((2*n-1)*n)=O(n^2),空间复杂度为O(1)。*/
    public String longestPalindrome(String s) {
        if(s.length() < 2)
            return s;
        String res = s.substring(0,1);
        for(int i=0; i<s.length()-1; i++){
            String temp = process(s, i, i);
            if(res.length() < temp.length())
                res = temp;
            
            temp = process(s, i, i+1);
            if(res.length() < temp.length())
                res = temp;
        }
        return res;
    }
    public String process(String s,int left,int right){
        while(left>=0 && right<s.length() && s.charAt(left)==s.charAt(right)){
            left--;
            right++;
        }
        return s.substring(left+1,right);
    }
    
}
```

## 44 [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

> 给定一个字符串，在不重复字符的情况下找出最长子字符串的长度。
>
> 示例1:
>
> 输入:“abcabcbb”
> 输出:3
> 说明:答案是“abc”，长度为3。
> 示例2:
>
> 输入:“bbbbb”
> 输出:1
> 说明:答案是“b”，长度为1。
> 示例3:
>
> 输入:“pwwkew”
> 输出:3
> 说明:答案为“wke”，长度为3。
> 注意，答案必须是子字符串，“pwke”是子序列，而不是子字符串。

```java
class Solution {
    //法1
    public int lengthOfLongestSubstring(String s) {
        int max = 0;
        HashMap<Character,Integer> map = new HashMap<>();
        for(int i=0,j=0; i<s.length(); i++){
            if(map.containsKey(s.charAt(i))){
                j = Math.max(map.get(s.charAt(i)), j);
            }
            if(max < i-j+1){
                max = i-j+1;
            }
            map.put(s.charAt(i),i+1);
        }
        return max;
    }
    //法2  滑动窗口思想
    public int lengthOfLongestSubstring1(String s){
        HashSet<Character> set = new HashSet<>();
        int max=0,left=0,right=0;
        while(left<s.length() && right<s.length()){
            if(!set.contains(s.charAt(right))){
                set.add(s.charAt(right++));
                max = Math.max(max,set.size());
            }else{
                set.remove(s.charAt(left++));
            }
        }
        return max;
    }
}
```

## 45 [Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)

> 给定一个只包含字符'('和')'的字符串，找出最长有效(格式良好)括号子字符串的长度。
>
> 示例1:
>
> 输入:“()”
> 输出:2
> 说明:最长有效的括号子字符串是“()”
> 示例2:
>
> 输入:“)()())”
> 输出:4
> 说明:最长有效的括号子字符串是“()()”

```java
class Solution {
    public int longestValidParentheses(String s) {
        if(s==null || s.length()<2)
            return 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        int max = 0;
        for(int i=0; i<s.length(); i++){
            if(s.charAt(i) == '('){
                stack.push(i);//压入的是下标
            }else{
                stack.pop();
                if(stack.isEmpty()){
                    stack.push(i);
                }else{
                    max = Math.max(max, i-stack.peek());
                }
            }
        }
        return max;
    }   
}
```

## 46 [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

> 给定一棵二叉树，找出树中两个给定节点的最低公共祖先(LCA)。
>
> 根据Wikipedia对LCA的定义:“在两个节点p和q之间定义的最低公共祖先是T中同时具有p和q作为后代的最低节点(在这里，我们允许一个节点作为其自身的后代)。”
>
> 给出如下二叉树:root = [3,5,1,6,2,0,8,null,null,7,4]
>
> ![img](https://assets.leetcode.com/uploads/2018/12/14/binarytree.png)
>
> Example 1:
>
> ```
> Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
> Output: 3
> 说明:节点5和节点1的LCA为3。
> ```
>
> **Example 2:**
>
> ```
> Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
> Output: 5
> 说明:节点5和节点4的LCA为5，因为根据LCA定义，节点可以是自身的后代。
> ```
>
>  注意:
>
> 所有节点的值都是惟一的。
> p和q是不同的，两个值都存在于二叉树中。

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null || p.val==root.val || q.val==root.val)
            return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left!=null && right!=null)
            return root;
        return left!=null ? left : right;
    }
}
```

## 47 [Majority Element](https://leetcode.com/problems/majority-element/)

> 给定一个大小为n的数组，找到“多数”元素。“多数”元素即出现次数大于⌊n / 2⌋的元素。
>
> 您可以假设数组是非空的，并且“多数”元素始终存在于数组中。
>
> 示例1:
>
> 输入:[3、2、3)
> 输出:3
> 示例2:
>
> 输入:[2 2 1,1,1,2,2]
> 输出:2

```java
class Solution {
    public int majorityElement(int[] nums) {
        int cur = nums[0], count = 1;
        for(int i=1; i<nums.length; i++){
            if(nums[i] == cur){
                count++;
            }else{
                count--;
            }
            if(count == 0){
                cur = nums[i];
                count = 1;
            }
        }
        count = 0;
        for(int i=0; i<nums.length; i++){
            if(nums[i] == cur)
                count++;
        }
        if(count*2 > nums.length)
            return cur;
        else
            return -1;
    }
}
```

## 48 [Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)

> 给定一个包含0和1的二维二进制矩阵，找出只包含1的最大矩形并返回其面积。
>
> **Example:**
>
> ```
> Input:
> [
>   ["1","0","1","0","0"],
>   ["1","0","1","1","1"],
>   ["1","1","1","1","1"],
>   ["1","0","0","1","0"]
> ]
> Output: 6
> ```

```java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        
    }
}
```

## 49  [Maximal Square](https://leetcode.com/problems/maximal-square/)

> 给定一个包含0和1的二维二进制矩阵，找出只包含1的最大正方形并返回其面积。
>
> **Example:**
>
> ```
> Input: 
> 
> 1 0 1 0 0
> 1 0 1 1 1
> 1 1 1 1 1
> 1 0 0 1 0
> 
> Output: 4
> ```

```java
class Solution {
        public int maximalSquare(char[][] matrix) {
        if(matrix==null || matrix.length==0)
            return 0;
        int row = matrix.length, col = matrix[0].length;
        int[][] dp = new int[row][col];
        int max = 0;
        //初始化第一行
        for(int i=0; i<col; i++){
            dp[0][i] = matrix[0][i] - '0';
            max = Math.max(max,dp[0][i]);
        }
        //初始化第一列
        for(int i=0; i<row; i++){
            dp[i][0] = matrix[i][0] - '0';
            max = Math.max(max,dp[i][0]);
        }
        for(int i=1; i<row; i++){
            for(int j=1; j<col; j++){
                if(matrix[i][j] == '0'){
                    dp[i][j] = 0;
                }else{
                    int left = dp[i][j-1];//当前点的左边点
                    int top = dp[i-1][j];//上边点
                    int Ltop = dp[i-1][j-1];//左上点
                    if(left>0 && top>0 && Ltop>0){
                        dp[i][j] = Math.min(Math.min(left,top),Ltop)+1;
                    }else{
                        dp[i][j] = 1;
                    }
                    max = Math.max(max,dp[i][j]);
                }
            }
        }
        return max*max;
    }
}
```

## 50 [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

> 给定一个二叉树，求它的最大深度。
>
> 最大深度是从根节点到最远叶节点的最长路径上的节点数。
>
> 注意:叶子是没有子节点的节点。

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null)
            return 0;
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return Math.max(left,right)+1;
    }
}
```

## 52 [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)

> 给定一个整数数组，在一个数组(至少包含一个数字)中找到具有最大乘积的的子数组。
>
> 示例1:
>
> 输入:(2、3、-2、4)
> 输出:6
> 说明:[2,3]有最大的乘积6。
> 示例2:
>
> 输入(-2 0 -1):
> 输出:0
> 说明:结果不能是2，因为[-2，-1]不是子数组。

  ```java
class Solution {
    public int maxProduct(int[] nums) {
        if(nums==null || nums.length==0)
            return 0;
        int res=nums[0], preMax=nums[0], preMin=nums[0];
        int curMax, curMin;
        for(int i=1; i<nums.length; i++){
        	curMax = Math.max(Math.max(preMax*nums[i], preMin*nums[i]), nums[i]);
        	curMin = Math.min(Math.min(preMax*nums[i], preMin*nums[i]), nums[i]);
        	preMax = curMax;
        	preMin = curMin;
        	
        	res = Math.max(res, curMax);
    	}
    	return res;
    }
}
  ```

## 53  [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

> 给定整数数组，找到具有最大和的相邻子数组(至少包含一个数字)并返回其和
>
> **Example:**
>
> ```
> Input: [-2,1,-3,4,-1,2,1,-5,4],
> Output: 6
> Explanation: [4,-1,2,1] has the largest sum = 6.
> ```
>
> 跟进:
>
> 如果您已经找到了O(n)的解决方案，那么尝试使用分治方法编写另一个解决方案，这种方法更加微妙。

```java
class Solution {
    public int maxSubArray(int[] nums) {
        if(nums==null || nums.length==0)
            return 0;
        int max=nums[0],curMax=nums[0];
        for(int i=1; i<nums.length; i++){
            curMax = Math.max(nums[i],curMax+nums[i]);
            max = Math.max(max,curMax);
        }
        return max;
    }
}
```

## 54 [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)

> 有两个大小分别为m和n的排序数组nums1和nums2。
>
> 求两个排序数组的中值。总的运行时复杂度应该是O(log (m+n))。
>
> 您可以假设nums1和nums2不能同时为空。
>
> **Example 1:**
>
> ```
> nums1 = [1, 3]
> nums2 = [2]
> 
> The median is 2.0
> ```
>
> **Example 2:**
>
> ```
> nums1 = [1, 2]
> nums2 = [3, 4]
> 
> The median is (2 + 3)/2 = 2.5
> ```

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        
    }
}
```

## 55 [Merge Intervals](https://leetcode.com/problems/merge-intervals/)

> 给定一组区间，合并所有重叠的区间。
>
> 示例1:
>
> 输入:[[1,3],[2,6],[8 10],[15、18]]
> 输出:[[1,6],[8 10],[15、18]]
> 说明:由于区间[1,3]和[2,6]重叠，将它们合并到[1,6]中。
> 示例2:
>
> 输入:[[1,4],[4,5]]
> 输出:[[1,5]]
> 说明:区间[1,4]和[4,5]被认为是重叠的。

```java
class Interval {
	  int start;
	  int end;
	  Interval() { start = 0; end = 0; }
	  Interval(int s, int e) { start = s; end = e; } 
}
public class Merge {
	 public List<Interval> merge(List<Interval> intervals) {
		 if(intervals==null || intervals.size()<2)
             return intervals;
         List<Interval> res = new ArrayList<>();
         Collections.sort(intervals,new Comparator<Interval>(){
             public int compare(Interval o1,Interval o2){
                 return o1.start-o2.start;
             } 
         });
         int newStart = intervals.get(0).start;
         int newEnd = intervals.get(0).end;
         for(Interval tmp : intervals){
             if(tmp.start <= newEnd && tmp.end >= newEnd)
                 newEnd = tmp.end;
             if(tmp.start > newEnd){
                 res.add(new Interval(newStart,newEnd));
                 newStart = tmp.start;
                 newEnd = tmp.end;
             }
         }
         res.add(new Interval(newStart,newEnd));
         
         return res;	        

     }
}
```

## 56 [Merge Two Binary Trees](https://leetcode.com/problems/merge-two-binary-trees/)

> 假设有两棵二叉树，假设你用其中一棵树覆盖另一棵树，两棵树的一些节点是重叠的，而另一些不是。
>
> 您需要将它们合并到一个新的二叉树中。合并规则是，如果两个节点重叠，则将节点值累加为合并节点的新值。否则，NOT null节点将被用作新树的节点。
>
> **Example 1:**
>
> ```
> Input: 
> 	Tree 1                     Tree 2                  
>           1                         2                             
>          / \                       / \                            
>         3   2                     1   3                        
>        /                           \   \                      
>       5                             4   7                  
> Output: 
> Merged tree:
> 	     3
> 	    / \
> 	   4   5
> 	  / \   \ 
> 	 5   4   7
> ```

```java
class Solution {
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if(t1 == null || t2 == null)
            return t1==null ? t2 : t1;
        TreeNode node = new TreeNode(t1.val+t2.val);
        node.left = mergeTrees(t1.left,t2.left);
        node.right = mergeTrees(t1.right,t2.right);
        return node;
    }
}
```

## 57 [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

> 合并两个已排序的链表，并将其作为一个新列表返回。新列表应该通过将前两个列表的节点拼接在一起来创建。
>
> **Example:**
>
> ```
> Input: 1->2->4, 1->3->4
> Output: 1->1->2->3->4->4
> ```

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1==null || l2==null)
            return l1==null ? l2 : l1;
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while(l1!=null && l2!=null){
            if(l1.val <= l2.val){
                cur.next = l1;
                l1 = l1.next;
            }else{
                cur.next = l2;
                l2 = l2.next;
            }  
            cur = cur.next;
        }
        cur.next = l1==null ? l2 : l1;
        return dummy.next;
    }
}
```

## 58 [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

> 合并k排序链表并将其作为一个排序链表返回。分析并描述其复杂性。
>
> **Example:**
>
> ```
> Input:
> [
>   1->4->5,
>   1->3->4,
>   2->6
> ]
> Output: 1->1->2->3->4->4->5->6
> ```

```java
class Solution {
    /*两两合并，就是前两个先合并，合并好了再跟第三个，然后第四个直到第k个。思路是对的，但是效率不高,
      这里就需要用到分治法 Divide and Conquer Approach。简单来说就是不停的对半划分，比如k个链表先划分为合并两个k/2个链表的任务，再不停的往下划分，直到划分成只有一个或两个链表的任务，开始合并。举个例子来说比如合并6个链表，那么按照分治法，我们首先分别合并0和3，1和4，2和5。这样下一次只需合并3个链表，我们再合并1和3，最后和2合并就可以了。代码中的k是通过 (len+1)/2 计算的，这里为啥要加1呢，这是为了当len为奇数的时候，k能始终从后半段开始，比如当n=5时，那么此时k=3，则0和3合并，1和4合并，最中间的2空出来。当len是偶数的时候，加1也不会有影响，比如当n=4时，此时k=2，那么0和2合并，1和3合并
    */
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists==null || lists.length==0)
            return null;
        int len = lists.length;
        while(len > 1){
            int k = (len+1)/2;
            for(int i=0; i<len/2; i++){
                lists[i] = mergeTwoLists(lists[i],lists[i+k]);
            }
            len = k;
        }
        return lists[0];
    }
     public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1==null || l2==null)
            return l1==null ? l2 : l1;
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while(l1!=null && l2!=null){
            if(l1.val <= l2.val){
                cur.next = l1;
                l1 = l1.next;
            }else{
                cur.next = l2;
                l2 = l2.next;
            }  
            cur = cur.next;
        }
        cur.next = l1==null ? l2 : l1;
        return dummy.next;
    }
}
```

## 59 [Min Stack](https://leetcode.com/problems/min-stack/)

> 设计一个栈，支持push、pop、top和在常量时间内检索最小元素。

```java
class MinStack {

    Stack<Integer> dataStack;
    Stack<Integer> minStack;
    /** initialize your data structure here. */
    public MinStack() {
        dataStack = new Stack<>();
        minStack = new Stack<>();
    }
    
    public void push(int x) {
        if(minStack.isEmpty()){
            minStack.push(x);
        }else{
            int min = minStack.peek();
            minStack.push(Math.min(min,x));
        }
        dataStack.push(x);
    }
    
    public void pop() {
        if(dataStack.isEmpty()){
            throw new IllegalArgumentException("the stack is empty!");
        }else{
            dataStack.pop();
            minStack.pop();
        }
    }
    
    public int top() {
        return dataStack.peek();
    }
    
    public int getMin() {
        if(minStack.isEmpty()){
            throw new IllegalArgumentException("the stack is empty!");
        }
        return minStack.peek();
    }
}

```

## 60 [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)

> 给定一个m x n的非负数网格，找到一条从左上角到右下角的路径，使其路径上所有数字的和最小化。
>
> 注意:您只能在任何时间点向下或向右移动。
>
> **Example:**
>
> ```
> Input:
> [
>   [1,3,1],
>   [1,5,1],
>   [4,2,1]
> ]
> Output: 7
> Explanation: Because the path 1→3→1→1→1 minimizes the sum.
> ```

```java
//暴力解法无法ac：Time Limit Exceeded
class Solution {
    public int minPathSum(int[][] grid) {
        if(grid.length==0 || grid[0].length==0)
            return 0;
        return process(grid,0,0);
    }
    public int process(int[][] grid,int i,int j,int res){
        if(i==grid.length-1 && j==grid[0].lrngth-1)
            return grid[i][j];
        if(i == grid.length-1)
            return grid[i][j]+process(grid,i,j+1);
        if(j == grid[0].length-1)
            return grid[i][j]+process(grid,i+1,j);
        return grid[i][j]+Math.min(process(grid,i+1,j),process(grid,i,j+1));
    }
}
//因此只能用动态规划
class Solution {
    public int minPathSum(int[][] grid) {
       if(grid.length==0 || grid[0].length==0)
            return 0;
        int row = grid.length;
        int col = grid[0].length;
        int[][] dp = new int[row][col];
        dp[row-1][col-1] = grid[row-1][col-1];
        for(int i=row-2; i>=0; i--){
            dp[i][col-1] = dp[i+1][col-1]+grid[i][col-1];
        }
        for(int j=col-2; j>=0; j--){
            dp[row-1][j] = dp[row-1][j+1]+grid[row-1][j];
        }
        for(int i=row-2; i>=0; i--){
            for(int j=col-2; j>=0; j--){
                dp[i][j] = Math.min(dp[i+1][j],dp[i][j+1])+grid[i][j];
            }
        }
        return dp[0][0];
    }
}
```

## 61 [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

> 给定一个字符串S和一个字符串T，找出S中的最小窗口，该窗口将包含的T中的所有字符。复杂度为O(n)
>
> **Example:**
>
> ```
> Input: S = "ADOBECODEBANC", T = "ABC"
> Output: "BANC"
> ```
>
> 注意:
>
> 如果S中没有覆盖T中所有字符的窗口，则返回空字符串""。
> 如果存在这样的窗口，则可以保证在S中始终只有一个惟一的最小窗口。

```java
class Solution {
    public String minWindow(String s, String t) {
        String res = "";
        int[] smap = new int[128];
        int[] tmap = new int[128];
        
        for(char c : t.toCharArray()){
            tmap[c]++;
        }
        int start = 0, end = 0, count = 0, minLen=Integer.MAX_VALUE;
        while(end < s.length()){
            if(tmap[s.charAt(end)] != 0){
                if(smap[s.charAt(end)] < tmap[s.charAt(end)]){
                    count++;
                }
                smap[s.charAt(end)]++;
            }
            if(count == t.length()){
                while(tmap[s.charAt(start)]==0 || smap[s.charAt(start)]>tmap[s.charAt(start)]){
                    if(smap[s.charAt(start)] > tmap[s.charAt(start)]){
                        smap[s.charAt(start)]--;
                    }
                    start++;
                }
                if(minLen > end-start+1){
                    res = s.substring(start,end+1); 
                    minLen = end-start+1;
                }
            }
            end++;                                                                                           }
        return res;
    }
}
```

## 62 [Move Zeroes](https://leetcode.com/problems/move-zeroes/)

> 给定一个数组编号，编写一个函数将所有0移到数组末尾，同时保持非零元素的相对顺序。
>
> **Example:**
>
> ```
> Input: [0,1,0,3,12]
> Output: [1,3,12,0,0]
> ```
>
> 注意:
>
> 您必须在不复制数组的情况下就地执行此操作。
> 最小化操作的总数

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int noZeroIndex = 0;//下标0-noZeroIndex范围都是非零数
        for(int i=0; i<nums.length; i++){
            if(nums[i] != 0){
                nums[noZeroIndex] = nums[i];
                noZeroIndex++;
            }
        }
        for(int i=noZeroIndex; i<nums.length; i++){
            nums[i] = 0;//之后的数都置为0
        }
    }
}
```

## 63 [Next Permutation](https://leetcode.com/problems/next-permutation/)

> 实现next置换，它将数字重新排列到字典上的next更大的数字置换中。
>
> 如果这样的安排是不可能的，它必须重新安排它作为最低的可能的顺序(即，按升序排序)。
>
> 替换必须到位，并且只使用固定的额外内存。
>
> 这里有一些例子。输入在左边一列，相应的输出在右边一列。
>
> ```
> 1,2,3 → 1,3,2
> 3,2,1 → 1,2,3
> 1,1,5 → 1,5,1
> ```
>
> 题意就是求下一个排列顺序，由题目中给的例子可以看出来，如果给定数组是降序，则说明是全排列的最后一种情况，则下一个排列就是最初始情况

```java
“这道题是给定一个数组和一个排列，求下一个排列。算法上其实没有什么特别的地方，主要的问题是经常不是一见到这个题就能马上理清思路。下面我们用一个例子来说明，比如排列是(2,3,6,5,4,1)，求下一个排列的基本步骤是这样：
1) 先从后往前找到第一个不是依次增长的数，记录下位置p。比如例子中的3，对应的位置是1;
2) 接下来分两种情况：
    (1) 如果上面的数字都是依次增长的，那么说明这是最后一个排列，下一个就是第一个，其实把所有数字反转过来即可(比如(6,5,4,3,2,1)下一个是(1,2,3,4,5,6));
    (2) 否则，如果p存在，从p开始往后找，找找找，找到第一个比他小的数的前一个数，然后两个调换位置，比如例子中的4。调换位置后得到(2,4,6,5,3,1)。最后把p之后的所有数字倒序，比如例子中得到(2,4,1,3,5,6), 这个即是要求的下一个排列。
以上方法中，最坏情况需要扫描数组三次，所以时间复杂度是O(3*n)=O(n)，空间复杂度是O(1)。代码如下：
 public class Solution {
 2     //http://blog.csdn.net/linhuanmars/article/details/20434115
 3     /*
 4     假设数组大小为 n
 5         1.从后往前，找到第一个 A[i-1] < A[i]的。也就是第一个排列中的  6那个位置，可以看到A[i]到A[n-1]这些都是单调递减序列。
 6         2.从 A[n-1]到A[i]中找到一个比A[i-1]大的值（也就是说在A[n-1]到A[i]的值中找到比A[i-1]大的集合中的最小的一个值）
 7         3.交换 这两个值，并且把A[n-1]到A[i+1]排序，从小到大。
 8     */
 9     public void nextPermutation(int[] num) {  
10         if(num==null || num.length==0)  
11             return;  
12         int i = num.length-2;  
13         while(i>=0 && num[i]>=num[i+1])  
14             i--;
15         
16         if(i>=0){  
17             int j=i+1;  
18             while(j<num.length && num[j]>num[i])
19                 j++;
20             j--;  
21             swap(num,i,j);  
22         }  
23         reverse(num, i+1,num.length-1);  
24     }    
25     private void swap(int[] num, int i, int j){  
26         int tmp = num[i];  
27         num[i] = num[j];  
28         num[j] = tmp;  
29     }  
30     private void reverse(int[] num, int i, int j){  
31         while(i < j)  
32             swap(num, i++, j--);  
33     }
```

## 64 [Number of Islands](https://leetcode.com/problems/number-of-islands/)

> 给定一个“1”(陆地)和“0”(水域)的二维网格地图，计算岛屿的数量。岛屿被水环绕，通过水平或垂直连接相邻的陆地而形成。你可以假设网格的四个边都被水包围着。
>
> **Example 1:**
>
> ```
> Input:
> 11110
> 11010
> 11000
> 00000
> 
> Output: 1
> ```
>
> **Example 2:**
>
> ```
> Input:
> 11000
> 11000
> 00100
> 00011
> 
> Output: 3
> ```

```java
class Solution {
    public int numIslands(char[][] grid) {
        if(grid==null || grid.length==0 || grid[0].length==0)
            return 0;
        //此行若写在if前面，当grid为空数组时会出现java.lang.ArrayIndexOutOfBoundsException
        int res=0, row=grid.length, col=grid[0].length;
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                if(grid[i][j] == '1'){
                    res++;
                    process(grid,i,j,row,col);
                }
            }
        }
        return res;
    }
    public void process(char[][] grid,int i,int j,int row,int col){
        //行非法或列法非法或当前数为0或为2（即已访问过），直接return
        if(i<0 || i>=row || j<0 || j>=col || grid[i][j]!='1')
            return;
        grid[i][j] = '2';//将已经访问过的置换为'2'
        process(grid,i+1,j,row,col);
        process(grid,i-1,j,row,col);
        process(grid,i,j+1,row,col);
        process(grid,i,j-1,row,col);
    }
}
```

## 65 [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

> 给定一个单链表，判断它是否是回文。
>
> Example 1:
>
> ```
> Input: 1->2
> Output: false
> ```
>
> Example 2:
>
> ```
> Input: 1->2->2->1
> Output: true
> ```
>
> 跟进: 你能在O(n)时间和O(1)空间内做吗?

```java
class Solution {
    //时间O(n)，空间O(n）
    public boolean isPalindrome(ListNode head) {
        Stack<Integer> stack = new Stack<>();
        ListNode cur = head;
        int len = 0;
        while(cur != null){
            stack.push(cur.val);
            cur = cur.next;
            len++;
        }
        cur = head;
        for(int i=1; i<len/2+1; i++){
            if(cur.val == stack.pop()){
                cur = cur.next;
            }else{
                return false;
            }
        }
        return true;
    }
    //O(n)时间和O(1)空间
public boolean isPalindrome(ListNode head) {
   	ListNode fast = head;
    ListNode slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;
    }
    if (fast != null) {
        slow = slow.next;
    }
    slow = reverse(slow);
    fast = head;
    while (slow != null) {
        if (fast.val != slow.val) {
            return false;
        }
        fast = fast.next;
        slow = slow.next;
    }
    return true;
}

public ListNode reverse(ListNode head) {
    ListNode prev = null;
    while (head != null) {
        ListNode next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
}
```

## 66 [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

> 给定一个字符串，您的任务是计算该字符串中有多少回文子字符串。
>
> 具有不同起始索引或结束索引的子字符串被计算为不同的子字符串，即使它们由相同的字符组成。
>
> **Example 1:**
>
> ```
> Input: "abc"
> Output: 3
> Explanation: Three palindromic strings: "a", "b", "c".
> ```
>
> **Example 2:**
>
> ```
> Input: "aaa"
> Output: 6
> Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
> ```
>
> 注：输入字符串长度不会超过1000。

```java
class Solution {
    /*这道题可以用递归来做，而且思路非常的简单粗暴。就是以字符串中的每一个字符都当作回文串中间的位置，然后向两边扩散，每当成功匹配两个左右两个字符，结果res自增1，然后再比较下一对。注意回文字符串有奇数和偶数两种形式，如果是奇数长度，那么i位置就是中间那个字符的位置，所以我们左右两遍都从i开始遍历；如果是偶数长度的，那么i是最中间两个字符的左边那个，右边那个就是i+1，这样就能cover所有的情况*/
   int count = 0;
   public int countSubstrings(String s) {
        if(s==null)
            return 0;  
        for(int i=0; i<s.length(); i++){
            countPalindrome(s, i, i);
            countPalindrome(s, i, i+1);
        }
        return count;
    }
    public void countPalindrome(String s,int i,int j){
        while(i>=0 && j<s.length() && s.charAt(i)==s.charAt(j)){
            i--;
            j++;
            count++;
        }
    }
}
```

## 67 [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)

> 给定一个只包含正整数的非空数组，找出该数组是否可以划分为两个子集，使两个子集中的元素之和相等。
>
> 注意:每个数组元素都不超过100。数组大小不超过200。
>
> **Example 1:**
>
> ```
> Input: [1, 5, 11, 5]
> 
> Output: true
> 
> Explanation: The array can be partitioned as [1, 5, 5] and [11].
> ```

```java
class Solution {
    public boolean canPartition(int[] nums)  {
        if(nums.length == 0) 
            return true;
        int sum = 0;
        for(int i=0; i<nums.length; i++){
            sum += nums[i];
        }
        if(sum%2 == 1){
            return false;
        }
        int target = sum/2;
        boolean[] dp = new boolean[target+1];
        dp[0] = true;
        for(int i=1; i<=nums.length; i++){
            for(int j=target; j>=nums[i-1]; j--){
                dp[j] = dp[j] || dp[j-nums[i-1]];
            }
        }
        return dp[target];
    }
}
```

## 68 [Path Sum III](https://leetcode.com/problems/path-sum-iii/)

> 给定一个二叉树，其中每个节点包含一个整数值。
> 找出与给定值求和的路径数。
> 路径不需要从根节点或叶子节点开始或结束，但是它必须向下(只从父节点移动到子节点)。
> 树的节点不超过1,000个，值的范围在-1,000,000到1,000,000之间。
>
> **Example:**
>
> ```
> root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
> 
>       10
>      /  \
>     5   -3
>    / \    \
>   3   2   11
>  / \   \
> 3  -2   1
> 
> Return 3. The paths that sum to 8 are:
> 
> 1.  5 -> 3
> 2.  5 -> 2 -> 1
> 3. -3 -> 11
> ```

```java
class Solution {
    public int pathSum(TreeNode root, int sum) {
        if(root == null)
            return 0;
        return find(root,sum)+pathSum(root.left,sum)+pathSum(root.right,sum);
    }
    public int find(TreeNode root,int sum){
        int res = 0;
        if(root == null)
            return res;
        if(root.val == sum){
            res++;
        }
        res += find(root.left, sum-root.val);
        res += find(root.right, sum-root.val);
        return res;
    }
}
```

## 69 [Perfect Squares](https://leetcode.com/problems/perfect-squares/)

> 给定一个正整数n，找出与n求和的最小的完全平方数(例如，1、4、9、16、…)。
>
> **Example 1:**
>
> ```
> Input: n = 12
> Output: 3 
> Explanation: 12 = 4 + 4 + 4.
> ```
>
> **Example 2:**
>
> ```
> Input: n = 13
> Output: 2
> Explanation: 13 = 4 + 9.
> ```

```java
class Solution {
    public int numSquares(int n) {
        if(n <= 0)
            return 0;
        int[] dp = new int[n+1];
        dp[0] = 0;
        for(int i=1; i<=n; i++){
            int min = Integer.MAX_VALUE;
            for(int j=1; j*j<=i; j++){
                min = Math.min(min,dp[i-j*j]+1);
            }
            dp[i] = min;
        }
        return dp[n];
    }
}
```

## 70 [Permutations](https://leetcode.com/problems/permutations/)

> 给定一组不同的整数，返回所有可能的排列。
>
> **Example:**
>
> ```
> Input: [1,2,3]
> Output:
> [
>   [1,2,3],
>   [1,3,2],
>   [2,1,3],
>   [2,3,1],
>   [3,1,2],
>   [3,2,1]
> ]
> ```

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if(nums == null || nums.length==0)
            return res;
        boolean[] visited = new boolean[nums.length];
        ArrayList<Integer> list = new ArrayList<>();
        
        dfs(res, list, nums, visited);
        return res;
    }
    public void dfs(List<List<Integer>> res,ArrayList<Integer> list,int[] nums,boolean[] visited){
        if(list.size() == nums.length){
            res.add(new ArrayList<Integer>(list));
            return;
        }
        for(int i=0; i<nums.length; i++){
            if(!visited[i]){
                visited[i] = true;
                list.add(nums[i]);
                dfs(res, list, nums, visited);
                list.remove(list.size()-1);
                visited[i] = false;
            }
        }
    }
}
```















