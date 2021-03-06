// mousehook.cpp : 定义应用程序的入口点。
//

#include "stdafx.h"
#include "mousehook.h"
#include "stdio.h"

#define MAX_LOADSTRING 100

// 全局变量:
HINSTANCE hInst;								// 当前实例
TCHAR szTitle[MAX_LOADSTRING];					// 标题栏文本
TCHAR szWindowClass[MAX_LOADSTRING];			// 主窗口类名

// my define 变量
#define WM_HOOKMSG 60001
HWND g_hWnd = NULL;
HHOOK hook,hook2;
LRESULT CALLBACK kbdProc(int code, WPARAM wParam, LPARAM lParam);
POINT mspt = {0,0};
int mskey = 0;
int count = 0;
extern void OpenPort();
extern void ClosePort();
extern void SerialSend(int key, int posx, int posy);

// 此代码模块中包含的函数的前向声明:
ATOM				MyRegisterClass(HINSTANCE hInstance);
BOOL				InitInstance(HINSTANCE, int);
LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY _tWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPTSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

 	// TODO: 在此放置代码。
	MSG msg;
	HACCEL hAccelTable;

	// 初始化全局字符串
	LoadString(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadString(hInstance, IDC_MOUSEHOOK, szWindowClass, MAX_LOADSTRING);
	MyRegisterClass(hInstance);

	// 执行应用程序初始化:
	if (!InitInstance (hInstance, nCmdShow))
	{
		return FALSE;
	}

	hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_MOUSEHOOK));

	HINSTANCE Hinstance;
	Hinstance = GetModuleHandle(TEXT("user32.dll"));
	if (Hinstance == NULL)
	{
		MessageBox(NULL, TEXT("没找到啊！"), TEXT("没找到"), 0);
	}
	hook=SetWindowsHookEx(WH_KEYBOARD_LL, kbdProc, Hinstance, NULL);
	hook2 = SetWindowsHookEx(WH_MOUSE_LL, kbdProc, Hinstance, NULL);
	OpenPort();
	SetTimer(g_hWnd, 0x555, 20, NULL);

	// 主消息循环:
	while (GetMessage(&msg, NULL, 0, 0))
	{
		if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	KillTimer(g_hWnd, 0x555);
	ClosePort();
	UnhookWindowsHookEx(hook);
	UnhookWindowsHookEx(hook2);

	return (int) msg.wParam;
}



//
//  函数: MyRegisterClass()
//
//  目的: 注册窗口类。
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEX wcex;

	wcex.cbSize = sizeof(WNDCLASSEX);

	wcex.style			= CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc	= WndProc;
	wcex.cbClsExtra		= 0;
	wcex.cbWndExtra		= 0;
	wcex.hInstance		= hInstance;
	wcex.hIcon			= LoadIcon(hInstance, MAKEINTRESOURCE(IDI_MOUSEHOOK));
	wcex.hCursor		= LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground	= (HBRUSH)(COLOR_WINDOW+1);
	wcex.lpszMenuName	= MAKEINTRESOURCE(IDC_MOUSEHOOK);
	wcex.lpszClassName	= szWindowClass;
	wcex.hIconSm		= LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

	return RegisterClassEx(&wcex);
}

//
//   函数: InitInstance(HINSTANCE, int)
//
//   目的: 保存实例句柄并创建主窗口
//
//   注释:
//
//        在此函数中，我们在全局变量中保存实例句柄并
//        创建和显示主程序窗口。
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   HWND hWnd;

   hInst = hInstance; // 将实例句柄存储在全局变量中

   hWnd = CreateWindow(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, 320, 200, NULL, NULL, hInstance, NULL);

   if (!hWnd)
   {
      return FALSE;
   }

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

//
//  函数: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  目的: 处理主窗口的消息。
//
//  WM_COMMAND	- 处理应用程序菜单
//  WM_PAINT	- 绘制主窗口
//  WM_DESTROY	- 发送退出消息并返回
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int wmId, wmEvent;
	PAINTSTRUCT ps;
	HDC hdc;
	RECT rect;

	char sz[64];
	WCHAR wsz[64];
	PMSLLHOOKSTRUCT mouse = (PMSLLHOOKSTRUCT)lParam;
	g_hWnd = hWnd;

	switch (message)
	{
	case WM_COMMAND:
		wmId    = LOWORD(wParam);
		wmEvent = HIWORD(wParam);
		// 分析菜单选择:
		switch (wmId)
		{
		case IDM_ABOUT:
			DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
			break;
		case IDM_EXIT:
			DestroyWindow(hWnd);
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		break;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		// TODO: 在此添加任意绘图代码...
		EndPaint(hWnd, &ps);
		break;
	case WM_HOOKMSG:
		mspt = mouse->pt;
//		sprintf_s(sz, "%i,%i,%i\r\n", mskey, mspt.x, mspt.y);
//		MultiByteToWideChar(CP_ACP, 0, sz, 64, wsz, 64);
//		OutputDebugString(wsz);
		break;
	case WM_TIMER:
		if (wParam == 0x555) {
			SerialSend(mskey, mspt.x, mspt.y);
			count++;
			count&=0xff;
			sprintf_s(sz, "%i,%i,%i\n", count, mspt.x, mspt.y);
//			sprintf_s(sz, "%i,%i,%i\n", mskey, mspt.x, mspt.y);

			hdc = GetDC(hWnd);
			GetClientRect(hWnd, &rect);
			MultiByteToWideChar(CP_ACP, 0, sz, 64, wsz, 64);
			DrawText(hdc,wsz,strlen(sz),&rect,DT_SINGLELINE);
			ReleaseDC(hWnd,hdc);
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

// “关于”框的消息处理程序。
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG:
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}

LRESULT CALLBACK kbdProc(int code, WPARAM wParam, LPARAM lParam)
{
	if (code == HC_ACTION) {
	//if (wParam == WM_LBUTTONDOWN || wParam == WM_MOUSEMOVE || wParam == WM_RBUTTONDOWN || wParam == WM_RBUTTONUP || wParam == WM_LBUTTONUP)
	//{
	//	return 1;
	//}

//		PKBDLLHOOKSTRUCT param = (PKBDLLHOOKSTRUCT)lParam;
//		if (param->vkCode == VK_ESCAPE)
//		{
//			PostQuitMessage(0);
//		}

		if (g_hWnd != NULL) {
			switch (wParam)
			{
			case WM_MOUSEMOVE:
				::SendMessage(g_hWnd, WM_HOOKMSG, wParam, lParam);
				break;
			case WM_LBUTTONDOWN:
				mskey = 1;
				::SendMessage(g_hWnd, WM_HOOKMSG, wParam, lParam);
				break;
			case WM_LBUTTONUP:
				mskey = 0;
				::SendMessage(g_hWnd, WM_HOOKMSG, wParam, lParam);
				break;
			}
		}
	}
	return CallNextHookEx(hook, code, wParam, lParam);
}

