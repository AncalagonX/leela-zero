
Dim oFSO, oFol
Set oFSO = WScript.CreateObject("Scripting.FileSystemObject")
Set oFol = oFSO.GetFolder("C:\Users\Owner\Downloads\leela-zero-next\leela-zero-next\src\Ray")

Dim oFile
For Each oFile In oFol.Files
	If LCase(Right(oFile.Name, 4)) = ".cpp" Or LCase(Right(oFile.Name, 2)) = ".h" Then
		Process oFile.Path
	End If
Next

Sub Process(path)

Dim oStream
Set oStream = WScript.CreateObject("ADODB.Stream")
oStream.Open
oStream.Type = 2
oStream.Charset = "utf-8"
oStream.LoadFromFile path

sText = oStream.ReadText

oStream.Close

Set oText = oFSO.CreateTextFile(path)
For i = 1 To Len(sText)
	sChr = Mid(sText, i, 1)
	If AscW(sChr) > 0 And AscW(sChr) <= &H7F Then 
		oText.Write sChr
	End If
Next
oText.Close

End Sub