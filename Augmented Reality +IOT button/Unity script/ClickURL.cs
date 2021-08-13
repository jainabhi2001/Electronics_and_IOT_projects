using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Events;

public class ClickURL : MonoBehaviour
{
    public string url;
    public void open()
    {
        StartCoroutine(GetRequest(url));    //It gets the URL from the button and makes a request to the browser for that URL
    }

    IEnumerator GetRequest(string uri)
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get(uri))
        {
            // Request and wait for the desired page.
            yield return webRequest.SendWebRequest();

        }
    }
}
