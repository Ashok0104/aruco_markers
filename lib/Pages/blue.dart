// bluetooth_button.dart
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class BluetoothButton extends StatelessWidget {
  final String resultId;

  BluetoothButton({required this.resultId});

  Future<void> sendBluetoothRequest() async {
    // Replace with your FastAPI server URL
    final url = Uri.parse('http://192.168.4.80:8000/send_bluetooth/');

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'result_id': resultId}),
      );

      if (response.statusCode == 200) {
        // Success
        print('Bluetooth transmission successful');
        print('Data sent: ${json.decode(response.body)['data_sent']}');
      } else {
        // Handle errors
        print('Error: ${json.decode(response.body)['error']}');
      }
    } catch (e) {
      // Handle network errors
      print('Network error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: sendBluetoothRequest,
      child: Text('Send via Bluetooth'),
    );
  }
}
