// mousehook.cpp : ����Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "mousehook.h"
#include "stdio.h"

#define MAX_LOADSTRING 100

// ȫ�ֱ���:
HINSTANCE hInst;								// ��ǰʵ��
TCHAR szTitle[MAX_LOADSTRING];					// �������ı�
TCHAR szWindowClass[MAX_LOADSTRING];			// ����������

// my define ����
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

// �˴���ģ���а����ĺ�����ǰ������:
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

 	// TODO: �ڴ˷��ô��롣
	MSG msg;
	HACCEL hAccelTable;

	// ��ʼ��ȫ���ַ���
	LoadString(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadString(hInstance, IDC_MOUSEHOOK, szWindowClass, MAX_LOADSTRING);
	MyRegisterClass(hInstance);

	// ִ��Ӧ�ó����ʼ��:
	if (!InitInstance (hInstance, nCmdShow))
	{
		return FALSE;
	}

	hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_MOUSEHOOK));

	HINSTANCE Hinstance;
	Hinstance = GetModuleHandle(TEXT("user32.dll"));
	if (Hinstance == NULL)
	{
		MessageBox(NULL, TEXT("û�ҵ�����"), TEXT("û�ҵ�"), 0);
	}
	hook=SetWindowsHookEx(WH_KEYBOARD_LL, kbdProc, Hinstance, NULL);
	hook2 = SetWindowsHookEx(WH_MOUSE_LL, kbdProc, Hinstance, NULL);
	OpenPort();
	SetTimer(g_hWnd, 0x555, 20, NULL);

	// ����Ϣѭ��:
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
//  ����: MyRegisterClass()
//
//  Ŀ��: ע�ᴰ���ࡣ
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
//   ����: InitInstance(HINSTANCE, int)
//
//   Ŀ��: ����ʵ�����������������
//
//   ע��:
//
//        �ڴ˺����У�������ȫ�ֱ����б���ʵ�������
//        ��������ʾ�����򴰿ڡ�
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   HWND hWnd;

   hInst = hInstance; // ��ʵ������洢��ȫ�ֱ�����

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
//  ����: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  Ŀ��: ���������ڵ���Ϣ��
//
//  WM_COMMAND	- ����Ӧ�ó���˵�
//  WM_PAINT	- ����������
//  WM_DESTROY	- �����˳���Ϣ������
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
		// �����˵�ѡ��:
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
		// TODO: �ڴ���������ͼ����...
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

// �����ڡ������Ϣ�������
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

